// per-SM throughput of the SwiGLU element math: 8 warps/SM, 64 independent elements per thread
#include "sm100.cuh"
using namespace s100;
template <int OP>
__device__ void body(float* out, int iters) {
  float v[16];
#pragma unroll
  for (int k = 0; k < 16; ++k) v[k] = 0.001f * (threadIdx.x + k);
  const unsigned long long t0 = clock64();
  for (int it = 0; it < iters; ++it) {
#pragma unroll
    for (int k = 0; k < 16; ++k) {
      const float a = v[k];
      if (OP == 0) v[k] = ex2f(a) * 0.5f;                         // 1 MUFU
      if (OP == 1) v[k] = a * sigmoid_kit(a) * 0.999f;            // kit sigmoid: 2 MUFU
      if (OP == 2) v[k] = a * sigmoid_poly(a) * 0.999f;           // 1 MUFU + poly
      if (OP == 3) v[k] = fmaf(a, 0.999f, 0.0001f);               // 1 FMA
      if (OP == 4) v[k] = a * sigmoid_nr(a) * 0.999f;             // 1 MUFU + Newton rcp
      if (OP == 5) v[k] = a * ((k & 1) ? sigmoid_nr(a) : sigmoid_kit(a)) * 0.999f;   // half and half
    }
  }
  const unsigned long long t1 = clock64();
  float s = 0; for (int k = 0; k < 16; ++k) s += v[k];
  out[blockIdx.x * blockDim.x + threadIdx.x] = s + (float)(t1 - t0) * 1e-30f;
  if (threadIdx.x == 0) out[gridDim.x * blockDim.x + blockIdx.x] = (float)(t1 - t0);
}
extern "C" __global__ void op_ex2(float* o, int n) { body<0>(o, n); }
extern "C" __global__ void op_sig(float* o, int n) { body<1>(o, n); }
extern "C" __global__ void op_sigp(float* o, int n) { body<2>(o, n); }
extern "C" __global__ void op_fma(float* o, int n) { body<3>(o, n); }
extern "C" __global__ void op_signr(float* o, int n) { body<4>(o, n); }
extern "C" __global__ void op_sigmix(float* o, int n) { body<5>(o, n); }
