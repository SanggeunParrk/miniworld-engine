// opm_prologue.cuh -- OPM K_P: LayerNorm(msa) -> left / right projections (32 each) -> token mask, straight into the GEMM1 operand
// layout a, b [S][L][32] bf16 (token t = s L + i). LN affine folded on the host: W' = W diag(gamma), bias' = W beta.
// One warp = 16 tokens per step: msa rows staged in smem, ldmatrix A fragments, LN statistics from the fragments (quad sums),
// normalized fragments x W'^T on the tensor cores (W' fragments resident in registers), output staged back through the same smem
// for 16-byte coalesced stores. Plus opm_maskbits: bits [L][S/32] (bit s of token (s, i)) for the pair counts.
#pragma once
#include "sm80_common.cuh"

namespace a100 {

struct OpmPrologueParams {
  const __nv_bfloat16* msa;   // [T, 64]
  const uint8_t* mask;        // [T] or nullptr
  const __nv_bfloat16* w;     // [64 out, 64 in]: rows 0..31 left, 32..63 right, gamma folded
  const float* bias;          // [64] W beta
  __nv_bfloat16* a;           // [T, 32]
  __nv_bfloat16* b;           // [T, 32]
  int T;
  float eps;
};

constexpr int OPM_P_WARPS = 4;

__global__ void __launch_bounds__(OPM_P_WARPS * 32) opm_prologue_kernel(OpmPrologueParams p) {
  __shared__ __align__(128) uint8_t smem[OPM_P_WARPS][16 * 128];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int g = lane >> 2, q = lane & 3;
  const uint32_t sb = smem_u32(smem[warp]);

  // W' B fragments: n-tile nt (8 outputs), k-chunk kc (16 inputs): b0 = W'[8nt+g][16kc+2q..], b1 = [..][16kc+8+2q..]
  uint32_t wb[8][4][2];
#pragma unroll
  for (int nt = 0; nt < 8; ++nt)
#pragma unroll
    for (int kc = 0; kc < 4; ++kc) {
      const __nv_bfloat16* r = p.w + (8 * nt + g) * 64 + 16 * kc + 2 * q;
      wb[nt][kc][0] = *reinterpret_cast<const uint32_t*>(r);
      wb[nt][kc][1] = *reinterpret_cast<const uint32_t*>(r + 8);
    }
  float bias[8][2];
#pragma unroll
  for (int nt = 0; nt < 8; ++nt) { bias[nt][0] = p.bias[8 * nt + 2 * q]; bias[nt][1] = p.bias[8 * nt + 2 * q + 1]; }

  const int nsteps = p.T / 16;
  for (int step = blockIdx.x * OPM_P_WARPS + warp; step < nsteps; step += gridDim.x * OPM_P_WARPS) {
    const int t0 = step * 16;
    // stage 16 rows x 128 B (granule-swizzled): lane loads granules lane, lane+32, lane+64, lane+96 (row = idx / 8)
    const uint4* src = reinterpret_cast<const uint4*>(p.msa + (size_t)t0 * 64);
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const int idx = lane + 32 * k, row = idx >> 3, gr = idx & 7;
      sts128(sb + row * 128 + ((gr ^ (row & 7)) << 4), __ldg(src + idx));
    }
    __syncwarp();
    uint32_t af[4][4];
#pragma unroll
    for (int kc = 0; kc < 4; ++kc) {
      const int row = lane & 15, gr = 2 * kc + (lane >> 4);
      ldsm_x4(af[kc], sb + row * 128 + ((gr ^ (row & 7)) << 4));
    }
    // LN statistics: rows g (regs 0, 2) and g + 8 (regs 1, 3); two-pass variance in fp32
    float s0 = 0.f, s1 = 0.f;
#pragma unroll
    for (int kc = 0; kc < 4; ++kc) {
      s0 += bf16lo(af[kc][0]) + bf16hi(af[kc][0]) + bf16lo(af[kc][2]) + bf16hi(af[kc][2]);
      s1 += bf16lo(af[kc][1]) + bf16hi(af[kc][1]) + bf16lo(af[kc][3]) + bf16hi(af[kc][3]);
    }
    const float mu0 = quad_sum(s0) * (1.f / 64), mu1 = quad_sum(s1) * (1.f / 64);
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
    float m0 = 1.f, m1 = 1.f;
    if (p.mask) { m0 = p.mask[t0 + g] ? 1.f : 0.f; m1 = p.mask[t0 + g + 8] ? 1.f : 0.f; }
    __syncwarp();   // every lane's ldmatrix is done before the staging area is overwritten
    // out staging: [a | b] halves, each 16 rows x 64 B; n-tile nt -> half nt/4, granule (nt%4) (8 channels = 16 B)
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
      float c[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int kc = 0; kc < 4; ++kc) mma16816(c, af[kc], wb[nt][kc][0], wb[nt][kc][1]);
      const uint32_t base = sb + (nt >> 2) * 1024, gr = nt & 3;
      // 64 B rows: granule gr ^ ((row >> 1) & 3) keeps the 16 B granules of 8 consecutive rows in distinct bank groups
      sts32(base + g * 64 + ((gr ^ ((g >> 1) & 3)) << 4) + 4 * q,
            pack_bf16((c[0] + bias[nt][0]) * m0, (c[1] + bias[nt][1]) * m0));
      sts32(base + (g + 8) * 64 + ((gr ^ (((g + 8) >> 1) & 3)) << 4) + 4 * q,
            pack_bf16((c[2] + bias[nt][0]) * m1, (c[3] + bias[nt][1]) * m1));
    }
    __syncwarp();
    // 2 halves x 16 rows x 4 granules = 128 granules: lane stores 4 (half = k / 2)
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const int idx = lane + 32 * k, half = idx >> 6, row = (idx >> 2) & 15, gr = idx & 3;
      const uint4 v = lds128(sb + half * 1024 + row * 64 + ((gr ^ ((row >> 1) & 3)) << 4));
      __nv_bfloat16* dst = (half ? p.b : p.a) + (size_t)(t0 + row) * 32 + gr * 8;
      stg128(dst, v);
    }
    __syncwarp();
  }
}

// bits [L][S/32]: bit b of word w = mask of token (32 w + b, i). Thread per (w, i), i fastest (coalesced mask reads).
__global__ void opm_maskbits_kernel(const uint8_t* __restrict__ mask, uint32_t* __restrict__ bits, int S, int L) {
  const int nw = S / 32;
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= nw * L) return;
  const int w = idx / L, i = idx % L;
  uint32_t v = 0;
#pragma unroll 8
  for (int b = 0; b < 32; ++b) v |= (mask ? (uint32_t)(mask[(size_t)(32 * w + b) * L + i] != 0) : 1u) << b;
  bits[i * nw + w] = v;
}

}  // namespace a100
