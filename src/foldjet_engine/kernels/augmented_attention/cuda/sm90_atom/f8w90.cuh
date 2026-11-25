// f8w90.cuh — EXPERIMENTAL e4m3 products of the bf16 forward row kernels (pre90_f8.cu, post90_f8.cu; opt-in, never the default route).
//
// The row kernels' GEMMs run as wgmma m64nNk32 e4m3 x e4m3 -> fp32 with the A operand from registers, as in the bf16 kernels:
//   A  the C-fragment tile of the activation, scaled per row by a power of two (2^floor(log2(448 / amax_row)), from the exact row amax:
//      dynamic, no calibration, the e4m3 cast cannot saturate) and packed 4 values per register.
//   B  the weight, e4m3 scaled by a power of two per 64-row unit / matrix (host side, once per weight version), K-major in the 128-byte swizzle; its K axis
//      is permuted inside each 32-slot step so the A fragment is the accumulator pairs as they lie: slot s = 16 h + 4 t + 2 a + b of a k32
//      step holds logical column 16 h + 8 a + 2 t + b (row90.cuh: tile nt, element e of lane t is logical column 8 nt + 2 t + e).
//   epilogue  acc * (1 / row scale) * (1 / weight scale): exact powers of two, so the only roundings are the two e4m3 casts.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cuda_fp8.h>
#include "rms90.cuh"

namespace r90 {
constexpr int KSTEPS8 = 4;  // k32 steps over K = 128

// e4m3 bytes of (x0, x1, x2, x3), byte i = x_i (cvt.e4m3x2: the first source to the upper byte)
DEVI uint32_t e4x4(float x0, float x1, float x2, float x3) {
  uint16_t lo, hi;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(lo) : "f"(x1), "f"(x0));
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(hi) : "f"(x3), "f"(x2));
  return (uint32_t)lo | ((uint32_t)hi << 16);
}
// the largest power of two s with amax * s <= 448 (amax 0: 1)
// integer arithmetic on the exponent (no division): amax = m 2^e, m in [1, 2): 448 / amax = (1.75 / m) 2^(8 - e), so the power is 8 - e, one
// less when m > 1.75; clamped to the normal range (a zero or subnormal amax gives the largest scale, harmless: the row is ~0)
DEVI float pow2_for(float amax) {
  const int b = __float_as_int(amax);
  const int e = (b >> 23) & 0xff, below = (b & 0x7fffff) > 0x600000;
  const int sb = min(max(262 - e - below, 1), 253);  // biased exponent of the scale: 127 + 8 - (e - 127) - below (<= 253: 1 / s normal)
  return __int_as_float(sb << 23);
}
// 1 / s of a power of two s in the normal range, exact
DEVI float inv_pow2(float s) { return __int_as_float(0x7f000000 - __float_as_int(s)); }
// row scales of the NT n-tiles 0 .. NT - 1 of a tile: s0 (row g), s1 (row g + 8)
template <int NT>
DEVI void row_scales(const float (*f)[4], float& s0, float& s1) {
  float a0 = 0.f, a1 = 0.f;
#pragma unroll
  for (int nt = 0; nt < NT; ++nt) {
    a0 = fmaxf(a0, fmaxf(fabsf(f[nt][0]), fabsf(f[nt][1])));
    a1 = fmaxf(a1, fmaxf(fabsf(f[nt][2]), fabsf(f[nt][3])));
  }
  a0 = fmaxf(a0, __shfl_xor_sync(~0u, a0, 1));
  a0 = fmaxf(a0, __shfl_xor_sync(~0u, a0, 2));
  a1 = fmaxf(a1, __shfl_xor_sync(~0u, a1, 1));
  a1 = fmaxf(a1, __shfl_xor_sync(~0u, a1, 2));
  s0 = pow2_for(a0);
  s1 = pow2_for(a1);
}
// A fragment of k32 step ks: n-tiles 4 ks .. 4 ks + 3 (rows g / g + 8 scaled by s0 / s1)
DEVI void afrag8(const float (*f)[4], int ks, float s0, float s1, uint32_t* a) {
  const float* p = f[4 * ks];
  const float* q = f[4 * ks + 1];
  const float* r = f[4 * ks + 2];
  const float* u = f[4 * ks + 3];
  a[0] = e4x4(p[0] * s0, p[1] * s0, q[0] * s0, q[1] * s0);
  a[1] = e4x4(p[2] * s1, p[3] * s1, q[2] * s1, q[3] * s1);
  a[2] = e4x4(r[0] * s0, r[1] * s0, u[0] * s0, u[1] * s0);
  a[3] = e4x4(r[2] * s1, r[3] * s1, u[2] * s1, u[3] * s1);
}

#define F8_WG(NR, NREG, LIST)                                                                                                      \
  DEVI void wg_rs_e4m3_n##NR(const uint32_t* a, uint64_t db, float* d, int accumulate) {                                         \
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %" #NREG ", 0;\n"                                                               \
                 "wgmma.mma_async.sync.aligned.m64n" #NR "k32.f32.e4m3.e4m3 " LIST ", p, 1, 1;\n}\n"                              \
                 : F8_OUT##NR                                                                                                       \
                 : "r"(accumulate), "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(db)                                             \
                 : "memory");                                                                                                       \
  }
#define F8_OUT64                                                                                                                    \
  "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]),          \
      "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]),       \
      "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]),       \
      "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
#define F8_OUT128                                                                                                                   \
  F8_OUT64, "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), \
      "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]),       \
      "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]),       \
      "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
F8_WG(64, 32,
      "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, "
      "%27, %28, %29, %30, %31}, {%33, %34, %35, %36}, %37")
F8_WG(128, 64,
      "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, "
      "%27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, "
      "%52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%65, %66, %67, %68}, %69")
#undef F8_WG

// The A operand quantized once (16 registers for K = 128 instead of the 64 fp32 of the tile), reused by every product that reads it.
template <int KS>
DEVI void quant8(const float (*f)[4], float s0, float s1, uint32_t (*a)[4]) {
#pragma unroll
  for (int ks = 0; ks < KS; ++ks) afrag8(f, ks, s0, s1, a[ks]);
}
template <int NR, int KS>
DEVI void wg_gemm8q(float* d, const uint32_t (*a)[4], uint32_t w, int accumulate) {
  wg_fence();
#pragma unroll
  for (int ks = 0; ks < KS; ++ks) {
    const uint64_t desc = wdesc(w + ks * 32);
    if constexpr (NR == 64) wg_rs_e4m3_n64(a[ks], desc, d, accumulate | (ks > 0));
    else wg_rs_e4m3_n128(a[ks], desc, d, accumulate | (ks > 0));
  }
  wg_commit();
  wg_wait();
}

// cp.async a [rows, kbytes] e4m3 weight tile (global row stride ld bytes) into 128-byte swizzled rows (kbytes <= 128)
DEVI void load_w8(uint32_t dst, const uint8_t* src, int ld, int rows, int kbytes, int tid, int nthr) {
  const int cpr = kbytes / 16;
  for (int i = tid; i < rows * cpr; i += nthr) {
    const int r = i / cpr, c = i % cpr;
    cp16(dst + r * 128 + ((c ^ (r & 7)) << 4), src + (size_t)r * ld + c * 16);
  }
}
}  // namespace r90
