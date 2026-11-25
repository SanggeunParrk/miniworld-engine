// adaln_bwd_tf32.cu — the RMS-adaLN backward of the fp32 SWA atom block's two norms on sm_100a, after the GEMM that gives dy (the FFN's
// dy = [da | db] Wu, the projections' dx = dP W): the Triton kernels' elementwise tails, one CUDA kernel for both:
//   xh = x rstd, rstd = 1 / sqrt(mean(x^2) + eps);  dxh = dy (1 + scale);  dx_out = res + rstd (dxh - xh mean(dxh xh))
//   d shift = sum_aug dy,  d scale = sum_aug dy xh,  [d gate = sum_aug res gin   (the FFN's gate_f: res = dq2, gin = ffn)],
//   [d gate_a = sum_aug out att   (the out-projection's gate: out = this kernel's dq1, att = the saved raw out-projection; column att_col)]
// -> dmod columns out_col (shift), out_col + C (scale), out_col + 2 C (gate, only when gin != null); scale is read from the same column
// out_col + C of mod. (FFN: out_col = 3C, x = q1, res = dq2; qkvg: out_col = 0, x = q, res = dq1.)
// One block per (batch element, atom): its eight warps walk the A augments (row n = a B + b), a warp = one 128-channel row (lane = four
// channels, 512 contiguous bytes per load), the row sums by warp shuffles; the augment sums stay in registers and meet in shared memory,
// so every dmod element is written once, by one thread, in a fixed order: no atomics, bit-reproducible.
// SPDX-License-Identifier: Apache-2.0
#include <stdint.h>
#include <cuda_bf16.h>

constexpr int C = 128, MODW = 6 * C, NWARP = 8;

// programmatic dependent launch (round 3 part 4): wait for the kernel before (its outputs are this one's inputs), then let the next
// kernel start its prologue under this one (it waits again before it reads)
__device__ __forceinline__ void pdl_first() {
  asm volatile("griddepcontrol.wait;" ::: "memory");
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

// four channels as float4 from fp32 or bf16 rows (round 3 part 6: master training reads the bf16 x / dq2 and writes the bf16 dx in place)
__device__ __forceinline__ float4 ld4(const float* p) { return *reinterpret_cast<const float4*>(p); }
__device__ __forceinline__ float4 ld4(const __nv_bfloat16* p) {
  const uint2 u = *reinterpret_cast<const uint2*>(p);
  return make_float4(__uint_as_float(u.x << 16), __uint_as_float(u.x & 0xffff0000u), __uint_as_float(u.y << 16), __uint_as_float(u.y & 0xffff0000u));
}
// dy = the sum of ndy partials ndys elements apart, in partial order (round 3 part 6: the FFN norm's dy as ffn_bwd_gate_tf32's DYF partials;
// ndy = 1: the GEMM's dy as it is)
__device__ __forceinline__ float4 ldy(const float* p, int ndy, int ndys) {
  if (ndy >= 4 && ndy <= 8) {                                                        // r3-swa part 20: the (<= 8) partials loaded at once, then added in order
    float4 w[8];                                                           // (a load per add was eight L2 round trips in a row)
#pragma unroll
    for (int g = 0; g < 8; ++g)
      if (g < ndy) w[g] = *reinterpret_cast<const float4*>(p + (size_t)g * ndys);
    float4 v = w[0];
#pragma unroll
    for (int g = 1; g < 8; ++g)
      if (g < ndy) { v.x += w[g].x; v.y += w[g].y; v.z += w[g].z; v.w += w[g].w; }
    return v;
  }
  float4 v = *reinterpret_cast<const float4*>(p);
  for (int g = 1; g < ndy; ++g) {
    const float4 w = *reinterpret_cast<const float4*>(p + (size_t)g * ndys);
    v.x += w.x; v.y += w.y; v.z += w.z; v.w += w.w;
  }
  return v;
}
__device__ __forceinline__ void st4(float* p, float4 v) { *reinterpret_cast<float4*>(p) = v; }
__device__ __forceinline__ void st4(__nv_bfloat16* p, float4 v) {
  const __nv_bfloat162 a = __floats2bfloat162_rn(v.x, v.y), b = __floats2bfloat162_rn(v.z, v.w);
  *reinterpret_cast<uint2*>(p) = make_uint2(*reinterpret_cast<const uint32_t*>(&a), *reinterpret_cast<const uint32_t*>(&b));
}

template <typename TX, typename TR, typename TO>
__device__ __forceinline__ void adaln_bwd(const TX* __restrict__ X, const float* __restrict__ DY, const TR* __restrict__ RES,
                         const float* __restrict__ GIN, const float* __restrict__ MOD, TO* __restrict__ OUT,
                         float* __restrict__ DMOD, int S, int A, int Bn, float eps, int out_col,
                         const float* __restrict__ ATT, int att_col, int ndy, int ndys) {
  __shared__ float red[NWARP][4][C];
  pdl_first();
  const int bs = (int)blockIdx.x, b = bs / S, s = bs % S;                  // the modulation row of this atom: b S + s = bs
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const float4 sc4 = *reinterpret_cast<const float4*>(MOD + (size_t)bs * MODW + out_col + C + 4 * lane);
  float4 dsh = make_float4(0.f, 0.f, 0.f, 0.f), dsc = dsh, dga = dsh, dat = dsh;
  for (int a = warp; a < A; a += NWARP) {
    const size_t o = (((size_t)a * Bn + b) * S + s) * C + 4 * lane;
    const float4 q = ld4(X + o), dy = ldy(DY + o, ndy, ndys);
    const float4 d2 = ld4(RES + o);
    const float rstd = 1.f / sqrtf(warp_sum(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w) * (1.f / C) + eps);
    const float4 xh = make_float4(q.x * rstd, q.y * rstd, q.z * rstd, q.w * rstd);
    const float4 dxh = make_float4(dy.x * (1.f + sc4.x), dy.y * (1.f + sc4.y), dy.z * (1.f + sc4.z), dy.w * (1.f + sc4.w));
    const float m = warp_sum(dxh.x * xh.x + dxh.y * xh.y + dxh.z * xh.z + dxh.w * xh.w) * (1.f / C);
    float4 r;
    r.x = d2.x + rstd * (dxh.x - xh.x * m); r.y = d2.y + rstd * (dxh.y - xh.y * m);
    r.z = d2.z + rstd * (dxh.z - xh.z * m); r.w = d2.w + rstd * (dxh.w - xh.w * m);
    st4(OUT + o, r);
    dsh.x += dy.x; dsh.y += dy.y; dsh.z += dy.z; dsh.w += dy.w;
    dsc.x += dy.x * xh.x; dsc.y += dy.y * xh.y; dsc.z += dy.z * xh.z; dsc.w += dy.w * xh.w;
    if (GIN != nullptr) {
      const float4 ff = *reinterpret_cast<const float4*>(GIN + o);
      dga.x += d2.x * ff.x; dga.y += d2.y * ff.y; dga.z += d2.z * ff.z; dga.w += d2.w * ff.w;
    }
    if (ATT != nullptr) {
      const float4 at = *reinterpret_cast<const float4*>(ATT + o);
      dat.x += r.x * at.x; dat.y += r.y * at.y; dat.z += r.z * at.z; dat.w += r.w * at.w;
    }
  }
  *reinterpret_cast<float4*>(&red[warp][0][4 * lane]) = dsh;
  *reinterpret_cast<float4*>(&red[warp][1][4 * lane]) = dsc;
  *reinterpret_cast<float4*>(&red[warp][2][4 * lane]) = dga;
  *reinterpret_cast<float4*>(&red[warp][3][4 * lane]) = dat;
  __syncthreads();
  const int nm = GIN != nullptr ? 3 : 2;
  for (int i = threadIdx.x; i < nm * C; i += NWARP * 32) {                 // i = 128 m + c: shift, scale, gate
    float t = 0.f;
#pragma unroll
    for (int w = 0; w < NWARP; ++w) t += red[w][i / C][i % C];
    DMOD[(size_t)bs * MODW + out_col + i] = t;
  }
  if (ATT != nullptr) {
    for (int c = threadIdx.x; c < C; c += NWARP * 32) {
      float t = 0.f;
#pragma unroll
      for (int w = 0; w < NWARP; ++w) t += red[w][3][c];
      DMOD[(size_t)bs * MODW + att_col + c] = t;
    }
  }
}

// One augment per atom (A = 1; B200 2026-11-17, round 3 part 4): the same equations with one warp per row and eight rows per block, no
// exchange (an augment sum is its one term: the values the kernel above writes, 0 + term). The kernel above launches a block per atom
// whose other seven warps find no augment -- 8192 blocks for one sample of 8192 atoms took ~30 us, twice per block backward.
template <typename TX, typename TR, typename TO>
__device__ __forceinline__ void adaln_bwd1(const TX* __restrict__ X, const float* __restrict__ DY, const TR* __restrict__ RES,
                          const float* __restrict__ GIN, const float* __restrict__ MOD, TO* __restrict__ OUT,
                          float* __restrict__ DMOD, int R, float eps, int out_col, const float* __restrict__ ATT, int att_col, int ndy,
                                          int ndys) {
  pdl_first();
  const int row = (int)blockIdx.x * NWARP + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  if (row >= R) return;
  const size_t o = (size_t)row * C + 4 * lane, om = (size_t)row * MODW + 4 * lane;
  const float4 sc4 = *reinterpret_cast<const float4*>(MOD + om + out_col + C);
  const float4 q = ld4(X + o), dy = ldy(DY + o, ndy, ndys);
  const float4 d2 = ld4(RES + o);
  const float rstd = 1.f / sqrtf(warp_sum(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w) * (1.f / C) + eps);
  const float4 xh = make_float4(q.x * rstd, q.y * rstd, q.z * rstd, q.w * rstd);
  const float4 dxh = make_float4(dy.x * (1.f + sc4.x), dy.y * (1.f + sc4.y), dy.z * (1.f + sc4.z), dy.w * (1.f + sc4.w));
  const float m = warp_sum(dxh.x * xh.x + dxh.y * xh.y + dxh.z * xh.z + dxh.w * xh.w) * (1.f / C);
  float4 r;
  r.x = d2.x + rstd * (dxh.x - xh.x * m); r.y = d2.y + rstd * (dxh.y - xh.y * m);
  r.z = d2.z + rstd * (dxh.z - xh.z * m); r.w = d2.w + rstd * (dxh.w - xh.w * m);
  st4(OUT + o, r);
  *reinterpret_cast<float4*>(DMOD + om + out_col) = make_float4(0.f + dy.x, 0.f + dy.y, 0.f + dy.z, 0.f + dy.w);
  *reinterpret_cast<float4*>(DMOD + om + out_col + C) =
      make_float4(0.f + dy.x * xh.x, 0.f + dy.y * xh.y, 0.f + dy.z * xh.z, 0.f + dy.w * xh.w);
  if (GIN != nullptr) {
    const float4 ff = *reinterpret_cast<const float4*>(GIN + o);
    *reinterpret_cast<float4*>(DMOD + om + out_col + 2 * C) =
        make_float4(0.f + d2.x * ff.x, 0.f + d2.y * ff.y, 0.f + d2.z * ff.z, 0.f + d2.w * ff.w);
  }
  if (ATT != nullptr) {
    const float4 at = *reinterpret_cast<const float4*>(ATT + o);
    *reinterpret_cast<float4*>(DMOD + om + att_col) = make_float4(0.f + r.x * at.x, 0.f + r.y * at.y, 0.f + r.z * at.z, 0.f + r.w * at.w);
  }
}

// entry points: fp32 x / res / out (the fp32 block), res bf16 (the FFN norm of master training: res = the bf16 dq2), x and out bf16
// (the qkvg norm of master training: x = the bf16 block input, out = the bf16 dx)
#define ADALN_ENTRIES(SUF, TX, TR, TO)                                                                                                   \
  extern "C" __global__ void __launch_bounds__(NWARP * 32)                                                                              \
  swa_adaln_bwd_tf32##SUF##_sm100(const TX* __restrict__ X, const float* __restrict__ DY, const TR* __restrict__ RES,                    \
                                  const float* __restrict__ GIN, const float* __restrict__ MOD, TO* __restrict__ OUT,                     \
                                  float* __restrict__ DMOD, int S, int A, int Bn, float eps, int out_col, const float* __restrict__ ATT, \
                                  int att_col, int ndy, int ndys) {                                                               \
    adaln_bwd<TX, TR, TO>(X, DY, RES, GIN, MOD, OUT, DMOD, S, A, Bn, eps, out_col, ATT, att_col, ndy, ndys);                                        \
  }                                                                                                                                      \
  extern "C" __global__ void __launch_bounds__(NWARP * 32)                                                                              \
  swa_adaln_bwd1_tf32##SUF##_sm100(const TX* __restrict__ X, const float* __restrict__ DY, const TR* __restrict__ RES,                   \
                                   const float* __restrict__ GIN, const float* __restrict__ MOD, TO* __restrict__ OUT,                    \
                                   float* __restrict__ DMOD, int R, float eps, int out_col, const float* __restrict__ ATT, int att_col, \
                                   int ndy, int ndys) {                                                                           \
    adaln_bwd1<TX, TR, TO>(X, DY, RES, GIN, MOD, OUT, DMOD, R, eps, out_col, ATT, att_col, ndy, ndys);                                              \
  }
ADALN_ENTRIES(, float, float, float)
ADALN_ENTRIES(_r16, float, __nv_bfloat16, float)
ADALN_ENTRIES(_xo16, __nv_bfloat16, float, __nv_bfloat16)
