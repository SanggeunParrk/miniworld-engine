// the element stages in isolation: TMEM ld -> math -> TMEM st, repeated, no MMA. MODE 0: backward DX gate (both warpgroups split the
// 64 columns of a chunk, as in tbwd.cu); MODE 1: forward SwiGLU (two warpgroups alternate chunks, 64 columns each, as in tfwd.cu).
#include "sm100.cuh"
using namespace s100;
DEVI float gate_da(float g, float b, float s, float l) { return (g * b) * (s + l * (1.f - s)); }
template <int MODE>
__device__ void body(unsigned long long* out, int iters) {
  __shared__ uint32_t tm;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  if (warp == 0) { tmem_alloc(smem_u32(&tm), 512); tmem_relinquish(); }
  tc_fence_before(); __syncthreads(); tc_fence_after();
  const uint32_t lb = (uint32_t)(warp & 3) * 32, trow = tm + (lb << 16);
  const int wg = warp >> 2;                       // 0 or 1
  // seed TMEM with plausible values
  {
    uint32_t v[16];
    for (int k = 0; k < 16; ++k) v[k] = __float_as_uint(0.05f * (k - 8) + 0.001f * lane);
    for (int c = 0; c < 512; c += 16) if (wg == 0) tmem_st16(trow + c, v);
    tmem_wait_st();
  }
  tc_fence_before(); __syncthreads(); tc_fence_after();
  const unsigned long long t0 = clock64();
  for (int it = 0; it < iters; ++it) {
    if (MODE == 0 || MODE == 2) {
      const int s = it & 1, half = wg;
      uint32_t dh[32], av[32], bv[32];
      tmem_ld32(trow + 256 + half * 32, dh);
      tmem_ld32(trow + s * 128 + half * 32, av);
      tmem_ld32(trow + s * 128 + 64 + half * 32, bv);
      tmem_wait_ld();
      uint32_t da[16], db[16];
#pragma unroll
      for (int k = 0; k < 16; ++k) {
        const uint32_t gp = pack_bf16(__uint_as_float(dh[2 * k]), __uint_as_float(dh[2 * k + 1]));
        const float g0 = bf16lo(gp), g1 = bf16hi(gp);
        const float a0 = __uint_as_float(av[2 * k]), a1 = __uint_as_float(av[2 * k + 1]);
        const float b0 = __uint_as_float(bv[2 * k]), b1 = __uint_as_float(bv[2 * k + 1]);
        if (MODE == 0) {
          const float s0 = sigmoid_kit(a0), s1 = sigmoid_kit(a1), l0 = a0 * s0, l1 = a1 * s1;
          da[k] = pack_bf16(gate_da(g0, b0, s0, l0), gate_da(g1, b1, s1, l1));
          db[k] = pack_bf16(g0 * l0, g1 * l1);
        } else { da[k] = pack_bf16(g0 * a0, g1 * a1); db[k] = pack_bf16(b0, b1); }
      }
      tmem_st16(trow + 384 + half * 32, da);
      tmem_st16(trow + 384 + half * 32 + 16, db);
      tmem_wait_st();
    } else if (MODE == 3) {
      // pipelined gate: the 32 columns in two 16-column steps; step 1's loads are in flight while step 0 is computed
      const int s = it & 1, half = wg;
      uint32_t dh0[16], av0[16], bv0[16], dh1[16], av1[16], bv1[16];
      tmem_ld16(trow + 256 + half * 32, dh0);
      tmem_ld16(trow + s * 128 + half * 32, av0);
      tmem_ld16(trow + s * 128 + 64 + half * 32, bv0);
      tmem_wait_ld();
      tmem_ld16(trow + 256 + half * 32 + 16, dh1);
      tmem_ld16(trow + s * 128 + half * 32 + 16, av1);
      tmem_ld16(trow + s * 128 + 64 + half * 32 + 16, bv1);
      uint32_t da[16], db[16];
      auto step = [&](const uint32_t (&dh)[16], const uint32_t (&av)[16], const uint32_t (&bv)[16], int o) {
#pragma unroll
        for (int k = 0; k < 8; ++k) {
          const uint32_t gp = pack_bf16(__uint_as_float(dh[2 * k]), __uint_as_float(dh[2 * k + 1]));
          const float g0 = bf16lo(gp), g1 = bf16hi(gp);
          const float a0 = __uint_as_float(av[2 * k]), a1 = __uint_as_float(av[2 * k + 1]);
          const float b0 = __uint_as_float(bv[2 * k]), b1 = __uint_as_float(bv[2 * k + 1]);
          const float s0 = sigmoid_kit(a0), s1 = sigmoid_kit(a1), l0 = a0 * s0, l1 = a1 * s1;
          da[o + k] = pack_bf16(gate_da(g0, b0, s0, l0), gate_da(g1, b1, s1, l1));
          db[o + k] = pack_bf16(g0 * l0, g1 * l1);
        }
      };
      step(dh0, av0, bv0, 0);
      tmem_wait_ld();
      step(dh1, av1, bv1, 8);
      tmem_st16(trow + 384 + half * 32, da);
      tmem_st16(trow + 384 + half * 32 + 16, db);
      tmem_wait_st();
    } else {
      if ((it & 1) != wg) continue;               // alternate chunks between the warpgroups
      const int s = it & 1;
      uint32_t hp[32];
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        uint32_t a[32], b[32];
        tmem_ld32(trow + s * 128 + half * 32, a);
        tmem_ld32(trow + s * 128 + 64 + half * 32, b);
        tmem_wait_ld();
#pragma unroll
        for (int k = 0; k < 16; ++k) {
          const float a0 = __uint_as_float(a[2 * k]), a1 = __uint_as_float(a[2 * k + 1]);
          const float b0 = __uint_as_float(b[2 * k]), b1 = __uint_as_float(b[2 * k + 1]);
          hp[half * 16 + k] = pack_bf16(a0 * sigmoid_kit(a0) * b0, a1 * sigmoid_kit(a1) * b1);
        }
      }
      uint32_t h0[16], h1[16];
#pragma unroll
      for (int k = 0; k < 16; ++k) { h0[k] = hp[k]; h1[k] = hp[16 + k]; }
      tmem_st16(trow + 256 + s * 32, h0);
      tmem_st16(trow + 256 + s * 32 + 16, h1);
      tmem_wait_st();
    }
  }
  const unsigned long long t1 = clock64();
  if (threadIdx.x == 0) out[blockIdx.x] = t1 - t0;
  tc_fence_before(); __syncthreads();
  if (warp == 0) { tc_fence_after(); tmem_dealloc(tm, 512); }
}
extern "C" __global__ void __launch_bounds__(256, 1) gate_dx(unsigned long long* o, int n) { body<0>(o, n); }
extern "C" __global__ void __launch_bounds__(256, 1) swiglu_fwd(unsigned long long* o, int n) { body<1>(o, n); }
extern "C" __global__ void __launch_bounds__(256, 1) gate_ldst(unsigned long long* o, int n) { body<2>(o, n); }
extern "C" __global__ void __launch_bounds__(256, 1) gate_pipe(unsigned long long* o, int n) { body<3>(o, n); }
