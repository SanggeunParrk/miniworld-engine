// adaln_bwd_sm80.cuh -- the adaLN-modulated RMSNorm backward of a 16-row tile that ends the FFN dy kernel and the qkvg backward kernel (A100 / sm_80).  For a row with the input q (q1 of the FFN half,
// the block input of the first half), its modulation scale, the gradient dx of the normalised-and-modulated tensor (the f1-ordered accumulators ``acc`` of the kernel's last product) and the residual
// gradient dres:
//
//   rstd = 1 / sqrt(mean(q^2) + eps),  xh = q rstd,  dxh = dx (1 + scale)
//   dq = rn(dres + rstd (dxh - xh mean(dxh xh))),  d scale = dx xh,  d shift = dx          (the modulation gradients: ``bwd_rows_sm80.cuh``)
//
// Three passes over the 32 channels a thread holds of a row (rstd, the mean, the outputs) that recompute xh and dxh instead of holding them.
#pragma once
#include "bwd_rows_sm80.cuh"
#include "qkvg_fwd_sm80.cuh"     // rn

namespace sw80 {

DEVI float4 ldg_f4b(const float* p) { return __ldg(reinterpret_cast<const float4*>(p)); }

// acc[gq][j][2 hh + e]: dx of channel 32 gq + 8 q4 + 2 j + e of row g8 + 8 hh.  ``scale`` = the modulation's scale columns of the batch row start at mod + mrow 768 + scale_col; the shift gradient
// goes to dmod[.., shift_col + c], the scale gradient to dmod[.., shift_col + 128 + c].
template <int MODE>
DEVI void adaln_bwd_epilogue(const float (&acc)[4][4][4], const RowMap& rm, int lane, const __nv_bfloat16* q, const __nv_bfloat16* dres, const float* mod, int scale_col, __nv_bfloat16* dq_out,
                             float* dmod, int shift_col, int B, int S, float eps) {
  const int q4 = lane & 3;
  float res[4][2] = {{0.f, 0.f}, {0.f, 0.f}, {0.f, 0.f}, {0.f, 0.f}};         // MODE_HOIST: per group of 32 channels, this lane's two reduced values (d shift | d scale), summed over the row halves
#pragma unroll
  for (int hh = 0; hh < 2; ++hh) {
    uint4 uq[4];
    float ss = 0.f;
#pragma unroll
    for (int gq = 0; gq < 4; ++gq) {
      uq[gq] = rm.rok[hh] ? ldg128(q + (size_t)rm.rr[hh] * 128 + 32 * gq + 8 * q4) : make_uint4(0u, 0u, 0u, 0u);
      const float e[8] = {bf16lo(uq[gq].x), bf16hi(uq[gq].x), bf16lo(uq[gq].y), bf16hi(uq[gq].y), bf16lo(uq[gq].z), bf16hi(uq[gq].z), bf16lo(uq[gq].w), bf16hi(uq[gq].w)};
#pragma unroll
      for (int i = 0; i < 8; ++i) ss = fmaf(e[i], e[i], ss);
    }
    const float rstd = 1.f / sqrtf(quad_sum(ss) * (1.f / 128.f) + eps);
    const float* mp = mod + (size_t)rm.mrow[hh] * 768 + scale_col;
    float dot = 0.f;
#pragma unroll
    for (int gq = 0; gq < 4; ++gq) {
      const float4 s0 = ldg_f4b(mp + 32 * gq + 8 * q4), s1 = ldg_f4b(mp + 32 * gq + 8 * q4 + 4);
      const float sc[8] = {s0.x, s0.y, s0.z, s0.w, s1.x, s1.y, s1.z, s1.w};
      const float e[8] = {bf16lo(uq[gq].x), bf16hi(uq[gq].x), bf16lo(uq[gq].y), bf16hi(uq[gq].y), bf16lo(uq[gq].z), bf16hi(uq[gq].z), bf16lo(uq[gq].w), bf16hi(uq[gq].w)};
#pragma unroll
      for (int i = 0; i < 8; ++i) dot = fmaf(acc[gq][i >> 1][2 * hh + (i & 1)] * (1.f + sc[i]), e[i] * rstd, dot);
    }
    const float mean = quad_sum(dot) * (1.f / 128.f);
#pragma unroll
    for (int gq = 0; gq < 4; ++gq) {
      const float4 s0 = ldg_f4b(mp + 32 * gq + 8 * q4), s1 = ldg_f4b(mp + 32 * gq + 8 * q4 + 4);
      const float sc[8] = {s0.x, s0.y, s0.z, s0.w, s1.x, s1.y, s1.z, s1.w};
      const float e[8] = {bf16lo(uq[gq].x), bf16hi(uq[gq].x), bf16lo(uq[gq].y), bf16hi(uq[gq].y), bf16lo(uq[gq].z), bf16hi(uq[gq].z), bf16lo(uq[gq].w), bf16hi(uq[gq].w)};
      uint4 udk = make_uint4(0u, 0u, 0u, 0u);
      if (rm.rok[hh]) udk = ldg128(dres + (size_t)rm.rr[hh] * 128 + 32 * gq + 8 * q4);
      const float dr[8] = {bf16lo(udk.x), bf16hi(udk.x), bf16lo(udk.y), bf16hi(udk.y), bf16lo(udk.z), bf16hi(udk.z), bf16lo(udk.w), bf16hi(udk.w)};
      float o[8], v[16];
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const float xh = e[i] * rstd, dx = acc[gq][i >> 1][2 * hh + (i & 1)];
        o[i] = dr[i] + rstd * (dx * (1.f + sc[i]) - xh * mean);
        v[i] = dx;
        v[8 + i] = dx * xh;
      }
      if (rm.rok[hh]) stg128(dq_out + (size_t)rm.rr[hh] * 128 + 32 * gq + 8 * q4, make_uint4(pack_bf16(o[0], o[1]), pack_bf16(o[2], o[3]), pack_bf16(o[4], o[5]), pack_bf16(o[6], o[7])));
      if (MODE == MODE_SINGLE) {
        if (rm.rok[hh]) {
          float* dm = dmod + (size_t)rm.mrow[hh] * 768 + shift_col + 32 * gq + 8 * q4;
          *reinterpret_cast<float4*>(dm) = make_float4(v[0], v[1], v[2], v[3]);
          *reinterpret_cast<float4*>(dm + 4) = make_float4(v[4], v[5], v[6], v[7]);
          *reinterpret_cast<float4*>(dm + 128) = make_float4(v[8], v[9], v[10], v[11]);
          *reinterpret_cast<float4*>(dm + 132) = make_float4(v[12], v[13], v[14], v[15]);
        }
      } else {
        reduce_scatter_g8<16>(v, lane);
        res[gq][0] += v[0];
        res[gq][1] += v[1];
      }
    }
  }
  if (MODE == MODE_HOIST && rm.rok[0]) {
    const int b2 = (lane >> 2) & 1, c0 = 4 * ((lane >> 3) & 1) + 2 * ((lane >> 4) & 1);      // g8_base(lane, 16) = 8 b2 + c0: the group (shift | scale) and the channel pair of this lane
#pragma unroll
    for (int gq = 0; gq < 4; ++gq)
      *reinterpret_cast<float2*>(dmod + ((size_t)rm.blk * B * S + rm.mrow[0]) * 768 + shift_col + 128 * b2 + 32 * gq + 8 * q4 + c0) = make_float2(res[gq][0], res[gq][1]);
  }
}

}  // namespace sw80
