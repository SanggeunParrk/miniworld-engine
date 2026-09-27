// pwa_value.cuh -- PWA K_Y: v[h][j][s * 32 + d] = LN(msa[s, j, :]) . Wv'[32 h + d, :] + bv[32 h + d]  (gamma folded into Wv', bv = Wv beta),
// head-major: the B operand layout of the contraction (K = j, N = (s, d)). A warp step is 16 consecutive s at one j: every head's
// output is then one 1 KB contiguous piece. Wv' (256 x 128 B) is staged once per CTA (swizzled) and read with ldmatrix.
// The gate operand is NOT produced here: the main kernel recomputes LN from the msa tile it reads for the residual.
#pragma once
#include "sm80_common.cuh"

namespace a100 {

struct PwaValueParams {
  const __nv_bfloat16* msa;   // [S*L, 64] (token s L + j)
  const __nv_bfloat16* wv;    // [256, 64] gamma folded
  const float* bv;            // [256]
  __nv_bfloat16* v;           // [8][L][S*32]
  int S, L;
  float eps;
};

constexpr int PWA_V_WARPS = 4;
constexpr int PWA_V_SMEM = 256 * 128 + PWA_V_WARPS * 16 * 128;

__global__ void __launch_bounds__(PWA_V_WARPS * 32) pwa_value_kernel(PwaValueParams p) {
  extern __shared__ __align__(128) uint8_t smem[];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int g = lane >> 2, q = lane & 3;
  const uint32_t sw = smem_u32(smem), sb = sw + 256 * 128 + warp * 16 * 128;
  for (int idx = tid; idx < 256 * 8; idx += PWA_V_WARPS * 32) {
    const int row = idx >> 3, gr = idx & 7;
    cp_async16(sw + row * 128 + ((gr ^ (row & 7)) << 4), p.wv + row * 64 + gr * 8);
  }
  cp_async_commit();
  cp_async_wait<0>();
  __syncthreads();

  const int nsb = p.S / 16, steps = nsb * p.L;
  const size_t ldv = (size_t)p.S * 32;
  for (int step = blockIdx.x * PWA_V_WARPS + warp; step < steps; step += gridDim.x * PWA_V_WARPS) {
    const int j = step / nsb, s0 = (step % nsb) * 16;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const int idx = lane + 32 * k, row = idx >> 3, gr = idx & 7;
      const uint4* src = reinterpret_cast<const uint4*>(p.msa + ((size_t)(s0 + row) * p.L + j) * 64) + gr;
      sts128(sb + row * 128 + ((gr ^ (row & 7)) << 4), __ldg(src));
    }
    __syncwarp();
    uint32_t af[4][4];
#pragma unroll
    for (int kc = 0; kc < 4; ++kc) {
      const int row = lane & 15, gr = 2 * kc + (lane >> 4);
      ldsm_x4(af[kc], sb + row * 128 + ((gr ^ (row & 7)) << 4));
    }
    __syncwarp();
    float s0s = 0.f, s1s = 0.f;
#pragma unroll
    for (int kc = 0; kc < 4; ++kc) {
      s0s += bf16lo(af[kc][0]) + bf16hi(af[kc][0]) + bf16lo(af[kc][2]) + bf16hi(af[kc][2]);
      s1s += bf16lo(af[kc][1]) + bf16hi(af[kc][1]) + bf16lo(af[kc][3]) + bf16hi(af[kc][3]);
    }
    const float mu0 = quad_sum(s0s) * (1.f / 64), mu1 = quad_sum(s1s) * (1.f / 64);
    float v0 = 0.f, v1 = 0.f;
#pragma unroll
    for (int kc = 0; kc < 4; ++kc) {
      float d;
      d = bf16lo(af[kc][0]) - mu0; v0 += d * d; d = bf16hi(af[kc][0]) - mu0; v0 += d * d;
      d = bf16lo(af[kc][2]) - mu0; v0 += d * d; d = bf16hi(af[kc][2]) - mu0; v0 += d * d;
      d = bf16lo(af[kc][1]) - mu1; v1 += d * d; d = bf16hi(af[kc][1]) - mu1; v1 += d * d;
      d = bf16lo(af[kc][3]) - mu1; v1 += d * d; d = bf16hi(af[kc][3]) - mu1; v1 += d * d;
    }
    const float r0 = rsqrtf(quad_sum(v0) * (1.f / 64) + p.eps), r1 = rsqrtf(quad_sum(v1) * (1.f / 64) + p.eps);
#pragma unroll
    for (int kc = 0; kc < 4; ++kc) {
      af[kc][0] = pack_bf16((bf16lo(af[kc][0]) - mu0) * r0, (bf16hi(af[kc][0]) - mu0) * r0);
      af[kc][2] = pack_bf16((bf16lo(af[kc][2]) - mu0) * r0, (bf16hi(af[kc][2]) - mu0) * r0);
      af[kc][1] = pack_bf16((bf16lo(af[kc][1]) - mu1) * r1, (bf16hi(af[kc][1]) - mu1) * r1);
      af[kc][3] = pack_bf16((bf16lo(af[kc][3]) - mu1) * r1, (bf16hi(af[kc][3]) - mu1) * r1);
    }
#pragma unroll 2
    for (int h = 0; h < 8; ++h) {
      float c[4][4];
#pragma unroll
      for (int nt = 0; nt < 4; ++nt)
#pragma unroll
        for (int e = 0; e < 4; ++e) c[nt][e] = 0.f;
#pragma unroll
      for (int kc = 0; kc < 4; ++kc)
#pragma unroll
        for (int np = 0; np < 2; ++np) {
          uint32_t b[4];
          const int row = 32 * h + 16 * np + (lane & 7) + ((lane >> 4) << 3), gr = 2 * kc + ((lane >> 3) & 1);
          ldsm_x4(b, sw + row * 128 + ((gr ^ (row & 7)) << 4));
          mma16816(c[2 * np], af[kc], b[0], b[1]);
          mma16816(c[2 * np + 1], af[kc], b[2], b[3]);
        }
      __nv_bfloat16* dst = p.v + ((size_t)h * p.L + j) * ldv + (size_t)s0 * 32;
#pragma unroll
      for (int nt = 0; nt < 4; ++nt) {
        const int d = 8 * nt + 2 * q;
        const float b0 = __ldg(p.bv + 32 * h + d), b1 = __ldg(p.bv + 32 * h + d + 1);
        stg32(dst + g * 32 + d, pack_bf16(c[nt][0] + b0, c[nt][1] + b1));
        stg32(dst + (g + 8) * 32 + d, pack_bf16(c[nt][2] + b0, c[nt][3] + b1));
      }
    }
  }
}

}  // namespace a100
