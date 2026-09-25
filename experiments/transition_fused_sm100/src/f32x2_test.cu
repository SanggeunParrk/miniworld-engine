#include <stdint.h>
extern "C" __global__ void t(float* o, const float* a) {
  const int i = threadIdx.x;
  uint64_t x, y, z;
  asm("mov.b64 %0, {%1, %2};" : "=l"(x) : "f"(a[2 * i]), "f"(a[2 * i + 1]));
  asm("mov.b64 %0, {%1, %2};" : "=l"(y) : "f"(a[2 * i + 64]), "f"(a[2 * i + 65]));
  asm("fma.rn.f32x2 %0, %1, %2, %1;" : "=l"(z) : "l"(x), "l"(y));
  asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(z) : "l"(y));
  asm("add.rn.f32x2 %0, %0, %1;" : "+l"(z) : "l"(x));
  float lo, hi; asm("mov.b64 {%0, %1}, %2;" : "=f"(lo), "=f"(hi) : "l"(z));
  uint16_t q; asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(q) : "f"(hi), "f"(lo));
  o[2 * i] = lo; o[2 * i + 1] = hi; o[128 + i] = (float)q;
}
