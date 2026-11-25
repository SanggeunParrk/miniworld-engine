// wide_elt.cuh -- the element type of the wide TriMul's row kernels: a 16-byte vector is 8 bf16 or 4 fp32 (the TF32 path).  The kernels are templates over one of
// these traits; arithmetic is fp32 either way, every rounding to the storage type is the single one at a store.
#pragma once
#include <cuda_bf16.h>
#include "sm80_common.cuh"

namespace a100 {

struct Bf16 {
  using T = __nv_bfloat16;
  static constexpr int VEC = 8, BYTES = 2;
  static constexpr bool IS_F32 = false;
  DEVI static void unpack(uint4 v, float (&f)[8]) {
    const uint32_t w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
    for (int j = 0; j < 4; ++j) { f[2 * j] = bf16lo(w[j]); f[2 * j + 1] = bf16hi(w[j]); }
  }
  DEVI static uint4 pack(const float (&f)[8]) { return make_uint4(pack_bf16(f[0], f[1]), pack_bf16(f[2], f[3]), pack_bf16(f[4], f[5]), pack_bf16(f[6], f[7])); }
  DEVI static uint4 ones() { return make_uint4(0x3f803f80u, 0x3f803f80u, 0x3f803f80u, 0x3f803f80u); }
  DEVI static float to_f(T x) { return __bfloat162float(x); }
  DEVI static T from_f(float x) { return __float2bfloat16_rn(x); }
};

struct F32 {
  using T = float;
  static constexpr int VEC = 4, BYTES = 4;
  static constexpr bool IS_F32 = true;
  DEVI static void unpack(uint4 v, float (&f)[4]) { f[0] = __uint_as_float(v.x); f[1] = __uint_as_float(v.y); f[2] = __uint_as_float(v.z); f[3] = __uint_as_float(v.w); }
  DEVI static uint4 pack(const float (&f)[4]) { return make_uint4(__float_as_uint(f[0]), __float_as_uint(f[1]), __float_as_uint(f[2]), __float_as_uint(f[3])); }
  DEVI static uint4 ones() { return make_uint4(0x3f800000u, 0x3f800000u, 0x3f800000u, 0x3f800000u); }
  DEVI static float to_f(T x) { return x; }
  DEVI static T from_f(float x) { return x; }
};

// round-to-nearest to the 10-bit TF32 mantissa (the value stays an fp32 word with the low 13 bits zero): the weights the TF32 MMAs read and whose sums the LayerNorm
// fold subtracts
DEVI float tf32_round(float x) {
  uint32_t r;
  asm("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(r) : "f"(x));
  return __uint_as_float(r);
}

}  // namespace a100
