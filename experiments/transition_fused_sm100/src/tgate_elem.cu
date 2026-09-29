// tgate_elem.cu — the SwiGLU backward as an elementwise pass for the wide widths when the forward SAVED a and b (bf16; no fp32
// recompute): from dh = bf16(dy Ws) and the saved a, b
//   sig = rcp(1 + ex2(-a log2 e)), silu = a sig,  h = bf16(silu b),  dA = bf16((dh b)(sig + silu(1 - sig))),  dB = bf16(dh silu)
// h -> [M][H], dA | dB -> dab [M][2H]. 8 units (16 B) per thread access, grid-stride. SPDX-License-Identifier: Apache-2.0
#include <cuda_bf16.h>
#include <stdint.h>

__device__ __forceinline__ float lo(uint32_t v) { return __uint_as_float(v << 16); }
__device__ __forceinline__ float hi(uint32_t v) { return __uint_as_float(v & 0xffff0000u); }
__device__ __forceinline__ uint32_t pk(float a, float b) { uint32_t r; asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(b), "f"(a)); return r; }
__device__ __forceinline__ float ex2f(float x) { float y; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
__device__ __forceinline__ float rcpf(float x) { float y; asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
__device__ __forceinline__ float sigk(float a) { return rcpf(__fadd_rn(1.f, ex2f(__fmul_rn(-1.4426950408889634f, a)))); }

extern "C" __global__ void __launch_bounds__(256)
transition_gate_elem(const uint4* __restrict__ dh, const uint4* __restrict__ a, const uint4* __restrict__ b, uint4* __restrict__ h,
                     uint4* __restrict__ dab, long long M, int H) {
  const long long nv = M * (H / 8);                          // 16-byte vectors per [M][H] tensor
  const int hv = H / 8;
  for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < nv; i += (long long)gridDim.x * blockDim.x) {
    const uint4 g4 = __ldcs(dh + i), a4 = __ldcs(a + i), b4 = __ldcs(b + i);
    const uint32_t gw[4] = {g4.x, g4.y, g4.z, g4.w}, aw[4] = {a4.x, a4.y, a4.z, a4.w}, bw[4] = {b4.x, b4.y, b4.z, b4.w};
    uint32_t ho[4], ao[4], bo[4];
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const float g0 = lo(gw[e]), g1 = hi(gw[e]), a0 = lo(aw[e]), a1 = hi(aw[e]), b0 = lo(bw[e]), b1 = hi(bw[e]);
      const float s0 = sigk(a0), s1 = sigk(a1), l0 = a0 * s0, l1 = a1 * s1;
      ho[e] = pk(l0 * b0, l1 * b1);
      ao[e] = pk((g0 * b0) * (s0 + l0 * (1.f - s0)), (g1 * b1) * (s1 + l1 * (1.f - s1)));
      bo[e] = pk(g0 * l0, g1 * l1);
    }
    const long long row = i / hv, c = i % hv;
    h[i] = make_uint4(ho[0], ho[1], ho[2], ho[3]);
    dab[row * (2 * hv) + c] = make_uint4(ao[0], ao[1], ao[2], ao[3]);
    dab[row * (2 * hv) + hv + c] = make_uint4(bo[0], bo[1], bo[2], bo[3]);
  }
}
