// bo_tab_sm100.cu -- the bias-only token DiT inference runner's hoisted conditioning table in ONE pass (B200 only;
// runner.FusedBiasOnlyDiT._tables3 / _tables3b at one block). The two table GEMMs (cuBLAS, no bias) write their products straight into
// the table [T, 6 x 768] = (gate1, gate2 | s1, s2, sh1, sh2); this pass adds the biases and applies the sigmoid to the first NS columns
// in place -- what torch.addmm(bias) x 2 + torch.cat + a strided sigmoid_ did in four launches (two of them strided elementwise kernels:
// ~26 us of a 190 us step at r104 master L512 S5, 2026-11-17).
//   tab[t, j] = f(tab[t, j] + bias[j]),  f = sigmoid for j < NS, identity else
// The value in fp32, one rounding to the table's dtype (bf16: the product was rounded to bf16 by the GEMM first).
// SPDX-License-Identifier: Apache-2.0
#include <cstdint>
#include <cuda_bf16.h>

constexpr int TPB = 256;

__device__ __forceinline__ float sg(float x) { return 1.f / (1.f + expf(-x)); }

// four consecutive elements per thread; W a multiple of 4, NS a multiple of 4
extern "C" __global__ void __launch_bounds__(TPB) bo_tab_sm100(void* __restrict__ tab, const void* __restrict__ bias, int T, int W, int NS,
                                                               int f32) {
  const int w4 = W / 4;
  const long long i = (long long)blockIdx.x * TPB + threadIdx.x;
  if (i >= (long long)T * w4) return;
  const int c = (int)(i % w4) * 4;
  const long long e = i * 4;
  float v[4], b[4];
  if (f32) {
    const float4 x = *reinterpret_cast<const float4*>(static_cast<const float*>(tab) + e);
    const float4 y = *reinterpret_cast<const float4*>(static_cast<const float*>(bias) + c);
    v[0] = x.x; v[1] = x.y; v[2] = x.z; v[3] = x.w;
    b[0] = y.x; b[1] = y.y; b[2] = y.z; b[3] = y.w;
  } else {
    const uint2 x = *reinterpret_cast<const uint2*>(static_cast<const __nv_bfloat16*>(tab) + e);
    const uint2 y = *reinterpret_cast<const uint2*>(static_cast<const __nv_bfloat16*>(bias) + c);
    const float2 x0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&x.x)), x1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&x.y));
    const float2 y0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&y.x)), y1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&y.y));
    v[0] = x0.x; v[1] = x0.y; v[2] = x1.x; v[3] = x1.y;
    b[0] = y0.x; b[1] = y0.y; b[2] = y1.x; b[3] = y1.y;
  }
  const bool s = c < NS;
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    v[k] += b[k];
    if (s) v[k] = sg(v[k]);
  }
  if (f32) {
    *reinterpret_cast<float4*>(static_cast<float*>(tab) + e) = make_float4(v[0], v[1], v[2], v[3]);
  } else {
    const __nv_bfloat162 a = __floats2bfloat162_rn(v[0], v[1]), d = __floats2bfloat162_rn(v[2], v[3]);
    uint2 u;
    u.x = *reinterpret_cast<const uint32_t*>(&a);
    u.y = *reinterpret_cast<const uint32_t*>(&d);
    *reinterpret_cast<uint2*>(static_cast<__nv_bfloat16*>(tab) + e) = u;
  }
}
