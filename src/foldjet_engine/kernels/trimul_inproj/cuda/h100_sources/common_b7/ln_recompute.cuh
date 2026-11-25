// SPDX-License-Identifier: Apache-2.0
// Anthropic LN fragment, configurable independent affine load chains.
#pragma once
#ifndef RECOMP_LN_CHAINS
#define RECOMP_LN_CHAINS 2
#endif
template <int KS, bool SERIAL = false, int CLS = math::REF>
TMN_DEVI LnStats ln_recompute_fragment(uint32_t (&fa)[KS][4], const float* sGamma, const float* sBeta, int lane, float eps) {
  constexpr float invC = 1.f / (16 * KS);
  float meanA, meanB;
  if (CLS == math::TX) {
    float sA_ = 0.f, sB_ = 0.f;
#pragma unroll
    for (int ks = 0; ks < KS; ++ks) {
      sA_ += bf16lo(fa[ks][0]) + bf16hi(fa[ks][0]) + bf16lo(fa[ks][2]) + bf16hi(fa[ks][2]);
      sB_ += bf16lo(fa[ks][1]) + bf16hi(fa[ks][1]) + bf16lo(fa[ks][3]) + bf16hi(fa[ks][3]);
    }
    meanA = quad_sum(sA_) * invC; meanB = quad_sum(sB_) * invC;
  } else {                                              // the statement: group sums of the 4 values a k-step holds per row, balanced tree over k-steps, lane butterfly
    float gA[KS], gB[KS];
#pragma unroll
    for (int ks = 0; ks < KS; ++ks) {
      gA[ks] = __fadd_rn(__fadd_rn(bf16lo(fa[ks][0]), bf16hi(fa[ks][0])), __fadd_rn(bf16lo(fa[ks][2]), bf16hi(fa[ks][2])));
      gB[ks] = __fadd_rn(__fadd_rn(bf16lo(fa[ks][1]), bf16hi(fa[ks][1])), __fadd_rn(bf16lo(fa[ks][3]), bf16hi(fa[ks][3])));
    }
    meanA = math::ln_mean(quad_sum(math::tree_sum(gA)), invC); meanB = math::ln_mean(quad_sum(math::tree_sum(gB)), invC);
  }
#pragma unroll
  for (int ks = 0; ks < KS; ++ks) fence_regs(fa[ks]);   // re-derive the fp32 values in each pass (2 ALU ops) instead of keeping them live
  float rA, rB;
  if (CLS == math::TX) {
    float vA = 0.f, vB = 0.f;
#pragma unroll
    for (int ks = 0; ks < KS; ++ks) {
      float d;
      d = bf16lo(fa[ks][0]) - meanA; vA += d * d; d = bf16hi(fa[ks][0]) - meanA; vA += d * d;
      d = bf16lo(fa[ks][2]) - meanA; vA += d * d; d = bf16hi(fa[ks][2]) - meanA; vA += d * d;
      d = bf16lo(fa[ks][1]) - meanB; vB += d * d; d = bf16hi(fa[ks][1]) - meanB; vB += d * d;
      d = bf16lo(fa[ks][3]) - meanB; vB += d * d; d = bf16hi(fa[ks][3]) - meanB; vB += d * d;
    }
    rA = rsqrtf(quad_sum(vA) * invC + eps); rB = rsqrtf(quad_sum(vB) * invC + eps);
  } else {
    float gA[KS], gB[KS];
#pragma unroll
    for (int ks = 0; ks < KS; ++ks) {
      gA[ks] = __fadd_rn(math::ln_sq_acc(math::ln_sq(bf16lo(fa[ks][0]), meanA), bf16hi(fa[ks][0]), meanA), math::ln_sq_acc(math::ln_sq(bf16lo(fa[ks][2]), meanA), bf16hi(fa[ks][2]), meanA));
      gB[ks] = __fadd_rn(math::ln_sq_acc(math::ln_sq(bf16lo(fa[ks][1]), meanB), bf16hi(fa[ks][1]), meanB), math::ln_sq_acc(math::ln_sq(bf16lo(fa[ks][3]), meanB), bf16hi(fa[ks][3]), meanB));
    }
    rA = math::ln_rstd(quad_sum(math::tree_sum(gA)), invC, eps); rB = math::ln_rstd(quad_sum(math::tree_sum(gB)), invC, eps);
  }
  const float mrA = meanA * rA, mrB = meanB * rB; (void)mrA; (void)mrB;
#pragma unroll
  for (int ks = 0; ks < KS; ++ks) fence_regs(fa[ks]);
  uint32_t chain[RECOMP_LN_CHAINS] = {};                        // two interleaved chains: step ks waits for step ks-2 (two steps' loads in flight)
#pragma unroll
  for (int ks = 0; ks < KS; ++ks) {
    const uint32_t k0 = (uint32_t)(16 * ks + 2 * (lane & 3)) + (SERIAL ? zero_dep(chain[ks & (RECOMP_LN_CHAINS-1)]) : 0u);
    const float2 g0 = lds64f(smem_u32(sGamma) + 4u * k0), b0 = lds64f(smem_u32(sBeta) + 4u * k0);
    const float2 g1 = lds64f(smem_u32(sGamma) + 4u * k0 + 32u), b1 = lds64f(smem_u32(sBeta) + 4u * k0 + 32u);
    if (CLS == math::TX) {                             // y = x * (r g) + (b - mean r g)
      fa[ks][0] = pack_bf16(fmaf(bf16lo(fa[ks][0]), rA * g0.x, fmaf(-mrA, g0.x, b0.x)), fmaf(bf16hi(fa[ks][0]), rA * g0.y, fmaf(-mrA, g0.y, b0.y)));
      fa[ks][1] = pack_bf16(fmaf(bf16lo(fa[ks][1]), rB * g0.x, fmaf(-mrB, g0.x, b0.x)), fmaf(bf16hi(fa[ks][1]), rB * g0.y, fmaf(-mrB, g0.y, b0.y)));
      fa[ks][2] = pack_bf16(fmaf(bf16lo(fa[ks][2]), rA * g1.x, fmaf(-mrA, g1.x, b1.x)), fmaf(bf16hi(fa[ks][2]), rA * g1.y, fmaf(-mrA, g1.y, b1.y)));
      fa[ks][3] = pack_bf16(fmaf(bf16lo(fa[ks][3]), rB * g1.x, fmaf(-mrB, g1.x, b1.x)), fmaf(bf16hi(fa[ks][3]), rB * g1.y, fmaf(-mrB, g1.y, b1.y)));
    } else {                                             // y = fma((x - mean) r, g, b): the statement
      fa[ks][0] = pack_bf16(math::ln_affine(bf16lo(fa[ks][0]), meanA, rA, g0.x, b0.x), math::ln_affine(bf16hi(fa[ks][0]), meanA, rA, g0.y, b0.y));
      fa[ks][1] = pack_bf16(math::ln_affine(bf16lo(fa[ks][1]), meanB, rB, g0.x, b0.x), math::ln_affine(bf16hi(fa[ks][1]), meanB, rB, g0.y, b0.y));
      fa[ks][2] = pack_bf16(math::ln_affine(bf16lo(fa[ks][2]), meanA, rA, g1.x, b1.x), math::ln_affine(bf16hi(fa[ks][2]), meanA, rA, g1.y, b1.y));
      fa[ks][3] = pack_bf16(math::ln_affine(bf16lo(fa[ks][3]), meanB, rB, g1.x, b1.x), math::ln_affine(bf16hi(fa[ks][3]), meanB, rB, g1.y, b1.y));
    }
    chain[ks & (RECOMP_LN_CHAINS-1)] = fa[ks][0] ^ fa[ks][3];
  }
  return LnStats{meanA, rA, meanB, rB};
}


template <int KS, bool SERIAL = false, int CLS = math::REF>
TMN_DEVI LnStats ln_stats_only(uint32_t (&fa)[KS][4], const float* sGamma, const float* sBeta, int lane, float eps) {
  constexpr float invC = 1.f / (16 * KS);
  float meanA, meanB;
  if (CLS == math::TX) {
    float sA_ = 0.f, sB_ = 0.f;
#pragma unroll
    for (int ks = 0; ks < KS; ++ks) {
      sA_ += bf16lo(fa[ks][0]) + bf16hi(fa[ks][0]) + bf16lo(fa[ks][2]) + bf16hi(fa[ks][2]);
      sB_ += bf16lo(fa[ks][1]) + bf16hi(fa[ks][1]) + bf16lo(fa[ks][3]) + bf16hi(fa[ks][3]);
    }
    meanA = quad_sum(sA_) * invC; meanB = quad_sum(sB_) * invC;
  } else {                                              // the statement: group sums of the 4 values a k-step holds per row, balanced tree over k-steps, lane butterfly
    float gA[KS], gB[KS];
#pragma unroll
    for (int ks = 0; ks < KS; ++ks) {
      gA[ks] = __fadd_rn(__fadd_rn(bf16lo(fa[ks][0]), bf16hi(fa[ks][0])), __fadd_rn(bf16lo(fa[ks][2]), bf16hi(fa[ks][2])));
      gB[ks] = __fadd_rn(__fadd_rn(bf16lo(fa[ks][1]), bf16hi(fa[ks][1])), __fadd_rn(bf16lo(fa[ks][3]), bf16hi(fa[ks][3])));
    }
    meanA = math::ln_mean(quad_sum(math::tree_sum(gA)), invC); meanB = math::ln_mean(quad_sum(math::tree_sum(gB)), invC);
  }
#pragma unroll
  for (int ks = 0; ks < KS; ++ks) fence_regs(fa[ks]);   // re-derive the fp32 values in each pass (2 ALU ops) instead of keeping them live
  float rA, rB;
  if (CLS == math::TX) {
    float vA = 0.f, vB = 0.f;
#pragma unroll
    for (int ks = 0; ks < KS; ++ks) {
      float d;
      d = bf16lo(fa[ks][0]) - meanA; vA += d * d; d = bf16hi(fa[ks][0]) - meanA; vA += d * d;
      d = bf16lo(fa[ks][2]) - meanA; vA += d * d; d = bf16hi(fa[ks][2]) - meanA; vA += d * d;
      d = bf16lo(fa[ks][1]) - meanB; vB += d * d; d = bf16hi(fa[ks][1]) - meanB; vB += d * d;
      d = bf16lo(fa[ks][3]) - meanB; vB += d * d; d = bf16hi(fa[ks][3]) - meanB; vB += d * d;
    }
    rA = rsqrtf(quad_sum(vA) * invC + eps); rB = rsqrtf(quad_sum(vB) * invC + eps);
  } else {
    float gA[KS], gB[KS];
#pragma unroll
    for (int ks = 0; ks < KS; ++ks) {
      gA[ks] = __fadd_rn(math::ln_sq_acc(math::ln_sq(bf16lo(fa[ks][0]), meanA), bf16hi(fa[ks][0]), meanA), math::ln_sq_acc(math::ln_sq(bf16lo(fa[ks][2]), meanA), bf16hi(fa[ks][2]), meanA));
      gB[ks] = __fadd_rn(math::ln_sq_acc(math::ln_sq(bf16lo(fa[ks][1]), meanB), bf16hi(fa[ks][1]), meanB), math::ln_sq_acc(math::ln_sq(bf16lo(fa[ks][3]), meanB), bf16hi(fa[ks][3]), meanB));
    }
    rA = math::ln_rstd(quad_sum(math::tree_sum(gA)), invC, eps); rB = math::ln_rstd(quad_sum(math::tree_sum(gB)), invC, eps);
  }
  return LnStats{meanA, rA, meanB, rB};
}
