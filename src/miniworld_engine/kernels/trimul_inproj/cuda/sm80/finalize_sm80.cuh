// finalize_sm80.cuh -- the end of the TriMul backward in one launch: the input-projection weight gradients and the LayerNorm_in gradients.
//
// B7 joint (or B7src + B8) leaves fp32 partials: dwp [G][4 CH][128] (per source group, K1 row order) and part [P][2][128] (per consumer CTA:
// d gamma_in, d beta_in).  This kernel sums them in a fixed order (bit-identical replays), undoes the K1 row order and writes
//   d W_l, d W_lg, d W_r, d W_rg   [CH, 128] bf16 or fp32, through the strides of the parameter they belong to (the bidirectional module stores
//                                   them [in, out]: a gradient with the same strides is taken over by autograd, not copied),
//   d gamma_in, d beta_in          [128] bf16 or fp32 (the parameter's dtype).
// One warp per output row (4 CH rows) plus one warp per LayerNorm vector.  It replaces the sums, the row gathers and the dtype casts that
// were separate torch launches on the critical path after B7.
#pragma once
#include <cuda_bf16.h>
#include "sm80_common.cuh"

namespace a100 {

struct FinalizeParams {
  const float* dwp;               // [G][4 CH][128]
  const float* part;              // [P][2][128]
  void* w[4];                     // outputs (bf16 or fp32, one dtype for the four), in the order of the parameters: W_l, W_lg, W_r, W_rg ([CH, 128] through rs / ks)
  int rs[4], ks[4];               // element strides of each output: output channel, input channel
  void* ln[2];                    // d gamma_in, d beta_in [128]
  int w_f32;                      // 1: the four weight gradients are fp32 (fp32 master parameters), 0: bf16
  int ln_bf16;                    // 1: the LayerNorm gradients are bf16, 0: fp32
  int G, P, CH;
};

__global__ void __launch_bounds__(256) finalize_kernel(const FinalizeParams p) {
  const int item = (int)(blockIdx.x * 8 + (threadIdx.x >> 5)), lane = threadIdx.x & 31, CH = p.CH;
  if (item < 4 * CH) {
    // output (k, oc): k = 0 W_l, 1 W_lg, 2 W_r, 3 W_rg  ->  K1 row r = 64 step + 16 nw + n, plane channel 32 step + 8 nw + (n % 8), gate rows n < 8
    const int k = item / CH, oc = item - k * CH;
    const int ocf = oc + ((k >> 1) ? CH : 0);
    const int r = 64 * (ocf >> 5) + 16 * ((ocf & 31) >> 3) + ((k & 1) ? 0 : 8) + (ocf & 7);
    const float4* src = reinterpret_cast<const float4*>(p.dwp + (size_t)r * 128) + lane;
    const size_t gstride = (size_t)4 * CH * 128 / 4;                          // float4 elements between source groups
    float4 s = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll 4
    for (int g = 0; g < p.G; ++g) {
      const float4 v = __ldg(src + (size_t)g * gstride);
      s.x += v.x; s.y += v.y; s.z += v.z; s.w += v.w;
    }
    const size_t ks = (size_t)p.ks[k];
    if (p.w_f32) {                                                             // fp32 gradient, never rounded to bf16
      float* dstf = reinterpret_cast<float*>(p.w[k]) + (size_t)oc * p.rs[k] + (size_t)(4 * lane) * ks;
      if (ks == 1) {
        *reinterpret_cast<float4*>(dstf) = s;
      } else {
        dstf[0] = s.x; dstf[ks] = s.y; dstf[2 * ks] = s.z; dstf[3 * ks] = s.w;
      }
      return;
    }
    __nv_bfloat16* dst = reinterpret_cast<__nv_bfloat16*>(p.w[k]) + (size_t)oc * p.rs[k] + (size_t)(4 * lane) * ks;
    if (ks == 1) {                                                             // row-major: one 8 B store
      *reinterpret_cast<uint2*>(dst) = make_uint2(pack_bf16(s.x, s.y), pack_bf16(s.z, s.w));
    } else {                                                                   // [in, out] storage: the four input channels are ks apart
      dst[0] = __float2bfloat16_rn(s.x); dst[ks] = __float2bfloat16_rn(s.y);
      dst[2 * ks] = __float2bfloat16_rn(s.z); dst[3 * ks] = __float2bfloat16_rn(s.w);
    }
  } else if (item < 4 * CH + 2) {
    const int which = item - 4 * CH;                                           // 0: d gamma_in, 1: d beta_in
    const float4* src = reinterpret_cast<const float4*>(p.part + (size_t)which * 128) + lane;
    float4 s = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll 8
    for (int q = 0; q < p.P; ++q) {
      const float4 v = __ldg(src + (size_t)q * 64);                            // one partial = 2 x 128 floats = 64 float4
      s.x += v.x; s.y += v.y; s.z += v.z; s.w += v.w;
    }
    if (p.ln_bf16) {
      __nv_bfloat16* dst = reinterpret_cast<__nv_bfloat16*>(p.ln[which]) + 4 * lane;
      dst[0] = __float2bfloat16_rn(s.x); dst[1] = __float2bfloat16_rn(s.y); dst[2] = __float2bfloat16_rn(s.z); dst[3] = __float2bfloat16_rn(s.w);
    } else {
      reinterpret_cast<float4*>(p.ln[which])[lane] = s;
    }
  }
}

}  // namespace a100
