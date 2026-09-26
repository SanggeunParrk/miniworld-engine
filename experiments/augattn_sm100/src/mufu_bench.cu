// MUFU.EX2 throughput probe: every thread runs NCH independent ex2 chains of N steps; optional FFMA2 filler per ex2.
#include "sm100.cuh"
using namespace s100;
#ifndef NCH
#define NCH 8
#endif
#ifndef FILL
#define FILL 0
#endif
extern "C" __global__ void __launch_bounds__(1024, 1) mufu_bench(float* out, int n) {
  float x[NCH];
  f2 y = mk2(threadIdx.x * 1e-3f, 0.5f);
#pragma unroll
  for (int c = 0; c < NCH; ++c) x[c] = -1e-3f * (threadIdx.x + c);
  for (int i = 0; i < n; ++i) {
#pragma unroll
    for (int c = 0; c < NCH; ++c) {
      x[c] = ex2f(x[c]) - 1.0001f;
#pragma unroll
      for (int f = 0; f < FILL; ++f) y = fma2(y, mk2(0.999f, 0.999f), mk2(x[c], 1e-3f));
    }
  }
  float s = lo2(y) + hi2(y);
#pragma unroll
  for (int c = 0; c < NCH; ++c) s += x[c];
  if (s == 1234.f) out[0] = s;
}
