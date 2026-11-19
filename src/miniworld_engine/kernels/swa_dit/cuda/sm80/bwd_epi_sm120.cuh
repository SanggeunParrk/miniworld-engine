#pragma once
#include "ffn_bwd_sm80.cuh"
namespace sw80 {
// ``adaln_bwd_epilogue`` with its row operands (the residual q, the residual gradient) requested at the start of the tile (``EpiPre``), so the epilogue of a warp that has the SM nearly to itself (5120 rows
// = 320 tiles on 170 SMs) is arithmetic and stores only, and the modulation scale read once per channel group.  The statements and their order are ``adaln_bwd_epilogue``'s.
struct EpiPre {
  uint4 uq[2][4], ud[2][4];
};

DEVI void epi_prefetch(EpiPre& e, const RowMap& rm, const __nv_bfloat16* q, const __nv_bfloat16* dres, int q4) {
#pragma unroll
  for (int hh = 0; hh < 2; ++hh)
#pragma unroll
    for (int gq = 0; gq < 4; ++gq) {
      const size_t off = (size_t)rm.rr[hh] * 128 + 32 * gq + 8 * q4;
      e.uq[hh][gq] = rm.rok[hh] ? ldg128(q + off) : make_uint4(0u, 0u, 0u, 0u);
      e.ud[hh][gq] = rm.rok[hh] ? ldg128(dres + off) : make_uint4(0u, 0u, 0u, 0u);
    }
}

template <int DBG = 0>
DEVI void epi_run(const float (&acc)[4][4][4], const RowMap& rm, const EpiPre& e, int q4, const float* mod, int scale_col, __nv_bfloat16* dq_out, float* dmod, int shift_col, float eps) {
#pragma unroll
  for (int hh = 0; hh < 2; ++hh) {
    const float* mp = mod + (size_t)rm.mrow[hh] * 768 + scale_col;
    float4 sc4[4][2];
#pragma unroll
    for (int gq = 0; gq < 4; ++gq) { sc4[gq][0] = ldg_f4b(mp + 32 * gq + 8 * q4); sc4[gq][1] = ldg_f4b(mp + 32 * gq + 8 * q4 + 4); }
    float ss = 0.f;
#pragma unroll
    for (int gq = 0; gq < 4; ++gq) {
      const uint4 u = e.uq[hh][gq];
      const float v[8] = {bf16lo(u.x), bf16hi(u.x), bf16lo(u.y), bf16hi(u.y), bf16lo(u.z), bf16hi(u.z), bf16lo(u.w), bf16hi(u.w)};
#pragma unroll
      for (int i = 0; i < 8; ++i) ss = fmaf(v[i], v[i], ss);
    }
    const float rstd = 1.f / sqrtf(quad_sum(ss) * (1.f / 128.f) + eps);
    float dot = 0.f;
#pragma unroll
    for (int gq = 0; gq < 4; ++gq) {
      const uint4 u = e.uq[hh][gq];
      const float sc[8] = {sc4[gq][0].x, sc4[gq][0].y, sc4[gq][0].z, sc4[gq][0].w, sc4[gq][1].x, sc4[gq][1].y, sc4[gq][1].z, sc4[gq][1].w};
      const float v[8] = {bf16lo(u.x), bf16hi(u.x), bf16lo(u.y), bf16hi(u.y), bf16lo(u.z), bf16hi(u.z), bf16lo(u.w), bf16hi(u.w)};
#pragma unroll
      for (int i = 0; i < 8; ++i) dot = fmaf(acc[gq][i >> 1][2 * hh + (i & 1)] * (1.f + sc[i]), v[i] * rstd, dot);
    }
    const float mean = quad_sum(dot) * (1.f / 128.f);
#pragma unroll
    for (int gq = 0; gq < 4; ++gq) {
      const uint4 u = e.uq[hh][gq], ur = e.ud[hh][gq];
      const float sc[8] = {sc4[gq][0].x, sc4[gq][0].y, sc4[gq][0].z, sc4[gq][0].w, sc4[gq][1].x, sc4[gq][1].y, sc4[gq][1].z, sc4[gq][1].w};
      const float v[8] = {bf16lo(u.x), bf16hi(u.x), bf16lo(u.y), bf16hi(u.y), bf16lo(u.z), bf16hi(u.z), bf16lo(u.w), bf16hi(u.w)};
      const float dr[8] = {bf16lo(ur.x), bf16hi(ur.x), bf16lo(ur.y), bf16hi(ur.y), bf16lo(ur.z), bf16hi(ur.z), bf16lo(ur.w), bf16hi(ur.w)};
      float o[8], w[16];
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const float xh = v[i] * rstd, dx = acc[gq][i >> 1][2 * hh + (i & 1)];
        o[i] = dr[i] + rstd * (dx * (1.f + sc[i]) - xh * mean);
        w[i] = dx;
        w[8 + i] = dx * xh;
      }
      if (rm.rok[hh]) {
        stg128(dq_out + (size_t)rm.rr[hh] * 128 + 32 * gq + 8 * q4, make_uint4(pack_bf16(o[0], o[1]), pack_bf16(o[2], o[3]), pack_bf16(o[4], o[5]), pack_bf16(o[6], o[7])));
        float* dm = dmod + (size_t)rm.mrow[hh] * 768 + shift_col + 32 * gq + 8 * q4;
        if (DBG != 4) {
        *reinterpret_cast<float4*>(dm) = make_float4(w[0], w[1], w[2], w[3]);
        *reinterpret_cast<float4*>(dm + 4) = make_float4(w[4], w[5], w[6], w[7]);
        *reinterpret_cast<float4*>(dm + 128) = make_float4(w[8], w[9], w[10], w[11]);
        *reinterpret_cast<float4*>(dm + 132) = make_float4(w[12], w[13], w[14], w[15]);
        }
      }
    }
  }
}
}  // namespace sw80
