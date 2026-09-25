#include "sm100.cuh"
using namespace s100;
extern "C" __global__ void ex2_test(const float* x, float* poly, float* mufu, float* sp, float* sk, float* snr, float* rnr, float* rmu, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) { poly[i] = ex2_poly(x[i]); mufu[i] = ex2f(x[i]); sp[i] = sigmoid_poly(x[i]); sk[i] = sigmoid_kit(x[i]); snr[i] = sigmoid_nr(x[i]); rnr[i] = rcp_nr(x[i]); rmu[i] = rcpf(x[i]); }
}
