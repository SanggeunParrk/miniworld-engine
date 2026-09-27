// pwa_pair.cuh -- PWA K_Z: w[h, i, j] = softmax_j( LN(z[i, j, :]) . Wb[h]^T  masked by mask[j] ), bf16 [8][L][L].
// One CTA per pair row i. Warps walk 16-j groups: the [16 j][128] z tile is staged (swizzled 256 B rows), read as mma A fragments,
// normalized in registers (quad-shuffle statistics), and projected with ONE n8 tile (8 heads, gamma folded into Wb, B fragments
// resident). Logits go to smem [8][L] fp32; then warp h does the softmax of head h and stores the bf16 row.
// Masked keys get the bf16 lowest value like the module (an all-masked row becomes uniform, as in PyTorch).
#pragma once
#include "sm80_common.cuh"

namespace a100 {

struct PwaPairParams {
  const __nv_bfloat16* z;     // [L*L, 128]
  const uint8_t* mask;        // [L] or nullptr
  const __nv_bfloat16* wb;    // [8, 128] gamma folded
  const float* bb;            // [8] Wb beta
  __nv_bfloat16* w;           // [8, L, L]
  int L;
  float eps;
};

constexpr int PWA_Z_WARPS = 8;
inline int pwa_pair_smem(int L) { return 8 * L * 4 + PWA_Z_WARPS * 16 * 256; }

__global__ void __launch_bounds__(PWA_Z_WARPS * 32) pwa_pair_kernel(PwaPairParams p) {
  extern __shared__ __align__(128) uint8_t smem[];
  float* logit = reinterpret_cast<float*>(smem);
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int g = lane >> 2, q = lane & 3;
  const int L = p.L, i = blockIdx.x;
  const uint32_t sb = smem_u32(smem + 8 * L * 4) + warp * 16 * 256;

  uint32_t wb[8][2];
#pragma unroll
  for (int kc = 0; kc < 8; ++kc) {
    const __nv_bfloat16* r = p.wb + g * 128 + 16 * kc + 2 * q;
    wb[kc][0] = *reinterpret_cast<const uint32_t*>(r);
    wb[kc][1] = *reinterpret_cast<const uint32_t*>(r + 8);
  }
  const float bb0 = p.bb[2 * q], bb1 = p.bb[2 * q + 1];
  constexpr float NEG = -3.3895313892515355e38f;   // torch.finfo(torch.bfloat16).min

  for (int jg = warp; jg < L / 16; jg += PWA_Z_WARPS) {
    const int j0 = jg * 16;
    const uint4* src = reinterpret_cast<const uint4*>(p.z + ((size_t)i * L + j0) * 128);
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      const int idx = lane + 32 * k, row = idx >> 4, gr = idx & 15;
      sts128(sb + row * 256 + ((gr ^ (row & 7)) << 4), __ldg(src + idx));
    }
    __syncwarp();
    uint32_t af[8][4];
#pragma unroll
    for (int kc = 0; kc < 8; ++kc) {
      const int row = lane & 15, gr = 2 * kc + (lane >> 4);
      ldsm_x4(af[kc], sb + row * 256 + ((gr ^ (row & 7)) << 4));
    }
    __syncwarp();
    float s0 = 0.f, s1 = 0.f;
#pragma unroll
    for (int kc = 0; kc < 8; ++kc) {
      s0 += bf16lo(af[kc][0]) + bf16hi(af[kc][0]) + bf16lo(af[kc][2]) + bf16hi(af[kc][2]);
      s1 += bf16lo(af[kc][1]) + bf16hi(af[kc][1]) + bf16lo(af[kc][3]) + bf16hi(af[kc][3]);
    }
    const float mu0 = quad_sum(s0) * (1.f / 128), mu1 = quad_sum(s1) * (1.f / 128);
    float v0 = 0.f, v1 = 0.f;
#pragma unroll
    for (int kc = 0; kc < 8; ++kc) {
      float d;
      d = bf16lo(af[kc][0]) - mu0; v0 += d * d; d = bf16hi(af[kc][0]) - mu0; v0 += d * d;
      d = bf16lo(af[kc][2]) - mu0; v0 += d * d; d = bf16hi(af[kc][2]) - mu0; v0 += d * d;
      d = bf16lo(af[kc][1]) - mu1; v1 += d * d; d = bf16hi(af[kc][1]) - mu1; v1 += d * d;
      d = bf16lo(af[kc][3]) - mu1; v1 += d * d; d = bf16hi(af[kc][3]) - mu1; v1 += d * d;
    }
    const float r0 = rsqrtf(quad_sum(v0) * (1.f / 128) + p.eps), r1 = rsqrtf(quad_sum(v1) * (1.f / 128) + p.eps);
    float c[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int kc = 0; kc < 8; ++kc) {
      uint32_t a[4];
      a[0] = pack_bf16((bf16lo(af[kc][0]) - mu0) * r0, (bf16hi(af[kc][0]) - mu0) * r0);
      a[1] = pack_bf16((bf16lo(af[kc][1]) - mu1) * r1, (bf16hi(af[kc][1]) - mu1) * r1);
      a[2] = pack_bf16((bf16lo(af[kc][2]) - mu0) * r0, (bf16hi(af[kc][2]) - mu0) * r0);
      a[3] = pack_bf16((bf16lo(af[kc][3]) - mu1) * r1, (bf16hi(af[kc][3]) - mu1) * r1);
      mma16816(c, a, wb[kc][0], wb[kc][1]);
    }
    const bool k0 = !p.mask || p.mask[j0 + g], k1 = !p.mask || p.mask[j0 + g + 8];
    logit[(2 * q) * L + j0 + g] = k0 ? c[0] + bb0 : NEG;
    logit[(2 * q + 1) * L + j0 + g] = k0 ? c[1] + bb1 : NEG;
    logit[(2 * q) * L + j0 + g + 8] = k1 ? c[2] + bb0 : NEG;
    logit[(2 * q + 1) * L + j0 + g + 8] = k1 ? c[3] + bb1 : NEG;
  }
  __syncthreads();
  // softmax of head `warp` over j; lane owns pairs j = 2 (lane + 32 k)
  const float* lg = logit + warp * L;
  float mx = -INFINITY;
  for (int j = 2 * lane; j < L; j += 64) mx = fmaxf(mx, fmaxf(lg[j], lg[j + 1]));
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
  constexpr float LOG2E = 1.4426950408889634f;
  float sum = 0.f;
  for (int j = 2 * lane; j < L; j += 64) sum += exp2f((lg[j] - mx) * LOG2E) + exp2f((lg[j + 1] - mx) * LOG2E);
  const float inv = 1.f / warp_sum(sum);
  __nv_bfloat16* dst = p.w + ((size_t)warp * L + i) * L;
  for (int j = 2 * lane; j < L; j += 64)
    stg32(dst + j, pack_bf16(exp2f((lg[j] - mx) * LOG2E) * inv, exp2f((lg[j + 1] - mx) * LOG2E) * inv));
}

}  // namespace a100
