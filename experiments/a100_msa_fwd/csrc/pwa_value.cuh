// pwa_value.cuh -- PWA K_Y: v[h][j][s * 32 + d] = LN(msa[s, j, :]) . Wv'[32 h + d, :] + bv[32 h + d]  (gamma folded into Wv', bv = Wv beta),
// head-major: the B operand layout of the contraction (K = j, N = (s, d)). A warp step is 16 consecutive s at one j: every head's
// output is then one 1 KB contiguous piece. Wv' (256 x 128 B) is staged once per CTA (swizzled) and read with ldmatrix.
// The gate operand is NOT produced here: the main kernel recomputes LN from the msa tile it reads for the residual.
// v2: the next step's msa rows are loaded into registers before the current step computes (v1 stalled on them: long_scoreboard 6.8),
// and each head's 16 x 32 output goes through a 1 KB smem piece so the stores are 16 B and contiguous (v1: 4 B scattered).
// Key compaction (idx != nullptr): only the n valid keys are projected, v_c[h][k][..] for k < n_pad (row stride ldw), key k <- token
// j = idx[k]; the padding rows k in [n, n_pad) reuse idx[0] (finite values; their softmax weights are 0). n, n_pad live on the device.
#pragma once
#include "sm80_common.cuh"

namespace a100 {

struct PwaValueParams {
  const __nv_bfloat16* msa;   // [S*L, 64] (token s L + j)
  const __nv_bfloat16* wv;    // [256, 64] gamma folded
  const float* bv;            // [256]
  __nv_bfloat16* v;           // [8][L][S*32]
  __nv_bfloat16* y;           // [S*L, 64] LN(msa) without affine, or nullptr (the split path's gate operand)
  const int* idx;             // [L] valid keys (compaction) or nullptr (dense: v [8][L][S*32])
  const int* cnt;             // [2] n, n_pad (device)
  int ldw;                    // v_c row count per head (>= n_pad)
  int S, L;
  float eps;
};

#ifndef PWA_V_WARPS_
#define PWA_V_WARPS_ 4
#endif
constexpr int PWA_V_WARPS = PWA_V_WARPS_;
constexpr int PWA_V_SMEM = 256 * 128 + 256 * 4 + PWA_V_WARPS * (16 * 128 + 1024);   // Wv', bv, per-warp staging

__global__ void __launch_bounds__(PWA_V_WARPS * 32) pwa_value_kernel(PwaValueParams p) {
  extern __shared__ __align__(128) uint8_t smem[];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int g = lane >> 2, q = lane & 3;
  const uint32_t sw = smem_u32(smem), sb = sw + 256 * 128 + 256 * 4 + warp * (16 * 128 + 1024), so = sb + 16 * 128;
  float* sbias = reinterpret_cast<float*>(smem + 256 * 128);     // bv in smem: v2's per-head __ldg of it sat on the long scoreboard
  for (int idx = tid; idx < 256; idx += PWA_V_WARPS * 32) sbias[idx] = p.bv[idx];
  for (int idx = tid; idx < 256 * 8; idx += PWA_V_WARPS * 32) {
    const int row = idx >> 3, gr = idx & 7;
    cp_async16(sw + row * 128 + ((gr ^ (row & 7)) << 4), p.wv + row * 64 + gr * 8);
  }
  cp_async_commit();
  cp_async_wait<0>();
  __syncthreads();

  const int nrows = p.idx ? __ldg(p.cnt + 1) : p.L, nval = p.idx ? __ldg(p.cnt) : p.L;
  const int vrows = p.idx ? p.ldw : p.L;
  const int nsb = p.S / 16, steps = nsb * nrows;
  const size_t ldv = (size_t)p.S * 32;
#ifndef PWA_V_SMAJOR
  // step -> (s-block, j) with j fastest: a CTA's warps read 4 consecutive tokens of each msa row (512 B runs; -7% vs s-block fastest)
  auto decode = [&](int step, int& k, int& s0) { k = step % nrows; s0 = (step / nrows) * 16; };
#else
  auto decode = [&](int step, int& k, int& s0) { k = step / nsb; s0 = (step % nsb) * 16; };
#endif
  auto token = [&](int k) { return p.idx ? __ldg(p.idx + (k < nval ? k : 0)) : k; };
  auto fetch = [&](int step, uint4 (&x)[4]) {
    int k0, s0;
    decode(step, k0, s0);
    const int j = token(k0);
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const int idx = lane + 32 * k, row = idx >> 3, gr = idx & 7;
      x[k] = __ldg(reinterpret_cast<const uint4*>(p.msa + ((size_t)(s0 + row) * p.L + j) * 64) + gr);
    }
  };
  const int stride = gridDim.x * PWA_V_WARPS;
  int step = blockIdx.x * PWA_V_WARPS + warp;
  uint4 cur[4];
  if (step < steps) fetch(step, cur);
  for (; step < steps; step += stride) {
    int kv, s0;
    decode(step, kv, s0);
    const int j = token(kv);
    uint4 nxt[4];
    if (step + stride < steps) fetch(step + stride, nxt);
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const int idx = lane + 32 * k, row = idx >> 3, gr = idx & 7;
      sts128(sb + row * 128 + ((gr ^ (row & 7)) << 4), cur[k]);
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
    if (p.y) {       // normalized rows back through the staging area, then 128 B per token
#pragma unroll
      for (int kc = 0; kc < 4; ++kc) {
        sts32(sb + g * 128 + (((2 * kc) ^ g) << 4) + 4 * q, af[kc][0]);
        sts32(sb + (g + 8) * 128 + (((2 * kc) ^ g) << 4) + 4 * q, af[kc][1]);
        sts32(sb + g * 128 + (((2 * kc + 1) ^ g) << 4) + 4 * q, af[kc][2]);
        sts32(sb + (g + 8) * 128 + (((2 * kc + 1) ^ g) << 4) + 4 * q, af[kc][3]);
      }
      __syncwarp();
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        const int idx = lane + 32 * k, row = idx >> 3, gr = idx & 7;
        stg128(p.y + ((size_t)(s0 + row) * p.L + j) * 64 + gr * 8, lds128(sb + row * 128 + ((gr ^ (row & 7)) << 4)));
      }
      __syncwarp();
    }
#pragma unroll 2
    for (int h = 0; h < 8; ++h) {
      float c[4][4];              // accumulators start at the bias (v3: a separate FADD pass waited on the bias loads)
#pragma unroll
      for (int nt = 0; nt < 4; ++nt) {
        const float2 bb = *reinterpret_cast<const float2*>(sbias + 32 * h + 8 * nt + 2 * q);
        c[nt][0] = c[nt][2] = bb.x;
        c[nt][1] = c[nt][3] = bb.y;
      }
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
      // stage the 16 x 32 piece (64 B rows, granule nt ^ ((row >> 1) & 3)), then 2 x 16 B per lane into the contiguous 1 KB
#pragma unroll
      for (int nt = 0; nt < 4; ++nt) {
        sts32(so + g * 64 + ((nt ^ ((g >> 1) & 3)) << 4) + 4 * q, pack_bf16(c[nt][0], c[nt][1]));
        sts32(so + (g + 8) * 64 + ((nt ^ (((g + 8) >> 1) & 3)) << 4) + 4 * q, pack_bf16(c[nt][2], c[nt][3]));
      }
      __syncwarp();
      __nv_bfloat16* dst = p.v + ((size_t)h * vrows + kv) * ldv + (size_t)s0 * 32;
#pragma unroll
      for (int k = 0; k < 2; ++k) {
        const int idx = lane + 32 * k, row = idx >> 2, gr = idx & 3;
        stg128(dst + row * 32 + gr * 8, lds128(so + row * 64 + ((gr ^ ((row >> 1) & 3)) << 4)));
      }
      __syncwarp();
    }
#pragma unroll
    for (int k = 0; k < 4; ++k) cur[k] = nxt[k];
  }
}

}  // namespace a100
