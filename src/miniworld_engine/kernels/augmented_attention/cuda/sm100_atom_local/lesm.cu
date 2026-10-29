// lesm.cu — the row kernels around ESMFold2's atom attention (``LocalDiTBlock(**ESMFOLD2)``, inference) on sm_100a (CUDA cores):
//
//   local_esm_bias     bias2[h][w][i][c] = LN(z[w, i, p - start_w]) gamma . Wb[h] log2(e) at the position p = 32 w - 64 + c (c < 160); -inf
//                      where p is not a key of query 32 w + i: outside [0, min(N, n_real)), masked, |p - (32 w + i)| > 64, or outside the
//                      window's 192-key subset (start_w: AlphaFold 3's slide of [32 w - 80, 32 w + 112) back inside the real atoms). The real
//                      atom count is the key mask's population count, taken per CTA on the device (no host read: CUDA-graph capturable).
//                      z: the trunked pair [nwin, 32, 192, 16] bf16 or fp32; a thread per (w, i, c), the four heads at once.
//   local_esm_qkrope   per head (4 x 32) RMSNorm of q and k (one weight [32] per tensor, eps in the denominator as torch's rms_norm), then
//                      the rotary with per-atom tables cos / sin [n, 16] (row r is atom r % n): channel c < 16 pairs with c + 16,
//                      (a, b) -> (a cos - b sin, b cos + a sin). A warp per row, 4 channels a lane (8 lanes a head; the partner is lane ^ 4).
//                      q, k [M, 128] (rows ld apart) bf16 -> qo, ko [M, 128] bf16.
// Training: local_esm_qkrope_bwd (+ local_esm_qk_fin), local_esm_silu_bwd (below); the pair bias's backward is lbias.cu's local_esm_bias_bwd.
// SPDX-License-Identifier: Apache-2.0
#include <cuda_bf16.h>
#include <cstdint>

#define DEVI __device__ __forceinline__
constexpr int KEYS = 192, HALF = 64, SPAN = 160, DM = 128;
constexpr float LOG2E = 1.4426950408889634f;

DEVI float param_f(const void* p, int i, int is_bf16) {
  return is_bf16 ? __uint_as_float((uint32_t)reinterpret_cast<const unsigned short*>(p)[i] << 16) : reinterpret_cast<const float*>(p)[i];
}

// pflags: bit 0 Wb bf16, bit 1 gamma bf16, bit 2 z bf16
extern "C" __global__ void __launch_bounds__(256)
local_esm_bias(const void* __restrict__ z, const void* __restrict__ gamma, const void* __restrict__ wb, const uint8_t* __restrict__ km,
               float* __restrict__ out, int N, int n0, float eps, int pflags) {
  __shared__ float w[4][16];
  __shared__ int red[8];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  if (tid < 64) w[tid >> 4][tid & 15] = param_f(wb, tid, pflags & 1) * param_f(gamma, tid & 15, pflags & 2) * LOG2E;
  int nr = n0;
  if (km != nullptr) {
    int c = 0;
    for (int i = 16 * tid; i < N; i += 16 * blockDim.x) {           // N is a multiple of 128: whole 16-byte groups
      const uint4 b = *reinterpret_cast<const uint4*>(km + i);
      const uint32_t w4[4] = {b.x, b.y, b.z, b.w};
#pragma unroll
      for (int j = 0; j < 4; ++j)
        c += ((w4[j] & 0xffu) != 0) + ((w4[j] & 0xff00u) != 0) + ((w4[j] & 0xff0000u) != 0) + ((w4[j] >> 24) != 0);
    }
#pragma unroll
    for (int off = 16; off >= 1; off >>= 1) c += __shfl_xor_sync(0xffffffffu, c, off);
    if (lane == 0) red[warp] = c;
  }
  __syncthreads();
  if (km != nullptr) {
    nr = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) nr += red[i];
  }
  const int nwin = N / 32;
  const size_t count = (size_t)nwin * 32 * SPAN, plane = count;
  for (size_t e = (size_t)blockIdx.x * blockDim.x + tid; e < count; e += (size_t)gridDim.x * blockDim.x) {
    const int c = (int)(e % SPAN), i = (int)((e / SPAN) % 32), wi = (int)(e / (32 * SPAN));
    const int p = 32 * wi - HALF + c, qp = 32 * wi + i;
    const int first = 32 * wi - (KEYS - 32) / 2;
    const int start = first < 0 ? 0 : min(first, nr - KEYS);         // AF3's slide (below 192 real atoms it may begin before atom 0)
    const int slot = p - start;
    const bool ok = p >= 0 && p < N && p < nr && (km == nullptr || km[p]) && abs(p - qp) <= HALF && slot >= 0 && slot < KEYS;
    float s[4] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY};
    if (ok) {
      const size_t r = ((size_t)wi * 32 + i) * KEYS + slot;
      float x[16];
      if (pflags & 4) {
        const uint4* q = reinterpret_cast<const uint4*>(reinterpret_cast<const __nv_bfloat16*>(z) + r * 16);
#pragma unroll
        for (int k = 0; k < 2; ++k) {
          const uint4 u = q[k];
          const uint32_t v[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
          for (int j = 0; j < 4; ++j) { x[8 * k + 2 * j] = __uint_as_float(v[j] << 16); x[8 * k + 2 * j + 1] = __uint_as_float(v[j] & 0xffff0000u); }
        }
      } else {
        const float4* q = reinterpret_cast<const float4*>(reinterpret_cast<const float*>(z) + r * 16);
#pragma unroll
        for (int k = 0; k < 4; ++k) { const float4 u = q[k]; x[4 * k] = u.x; x[4 * k + 1] = u.y; x[4 * k + 2] = u.z; x[4 * k + 3] = u.w; }
      }
      float mean = 0.f;
#pragma unroll
      for (int k = 0; k < 16; ++k) mean += x[k];
      mean *= 1.f / 16.f;
      float var = 0.f;
#pragma unroll
      for (int k = 0; k < 16; ++k) { x[k] -= mean; var += x[k] * x[k]; }
      const float rstd = rsqrtf(var * (1.f / 16.f) + eps);
#pragma unroll
      for (int hh = 0; hh < 4; ++hh) {
        float acc = 0.f;
#pragma unroll
        for (int k = 0; k < 16; ++k) acc = fmaf(x[k], w[hh][k], acc);
        s[hh] = acc * rstd;
      }
    }
#pragma unroll
    for (int hh = 0; hh < 4; ++hh) out[hh * plane + e] = s[hh];
  }
}

DEVI float4 ld4(const __nv_bfloat16* p) {
  const uint2 u = *reinterpret_cast<const uint2*>(p);
  return make_float4(__uint_as_float(u.x << 16), __uint_as_float(u.x & 0xffff0000u), __uint_as_float(u.y << 16), __uint_as_float(u.y & 0xffff0000u));
}
DEVI uint32_t pk(float lo, float hi) { uint32_t r; asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(r) : "f"(lo), "f"(hi)); return r; }

extern "C" __global__ void __launch_bounds__(256)
local_esm_qkrope(const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k, int ld, const float* __restrict__ wq,
                 const float* __restrict__ wk, const float* __restrict__ cs, const float* __restrict__ sn, __nv_bfloat16* __restrict__ qo,
                 __nv_bfloat16* __restrict__ ko, int M, int n, float eps) {
  const int row = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  if (row >= M) return;
  const int c = 4 * (lane & 7);                                      // channels c .. c + 3 of head lane / 8
  const bool lo = c < 16;
  const int atom = row % n;
  const float4 cv = *reinterpret_cast<const float4*>(cs + (size_t)atom * 16 + (c & 15));
  const float4 sv = *reinterpret_cast<const float4*>(sn + (size_t)atom * 16 + (c & 15));
#pragma unroll
  for (int t = 0; t < 2; ++t) {
    const __nv_bfloat16* src = t ? k : q;
    const float* wt = t ? wk : wq;
    float4 x = ld4(src + (size_t)row * ld + 4 * lane);
    float ss = x.x * x.x + x.y * x.y + x.z * x.z + x.w * x.w;
    ss += __shfl_xor_sync(0xffffffffu, ss, 1); ss += __shfl_xor_sync(0xffffffffu, ss, 2); ss += __shfl_xor_sync(0xffffffffu, ss, 4);
    const float r = rsqrtf(ss * (1.f / 32.f) + eps);
    const float4 wv = *reinterpret_cast<const float4*>(wt + c);
    x.x *= r * wv.x; x.y *= r * wv.y; x.z *= r * wv.z; x.w *= r * wv.w;
    float4 y;                                                        // the partner channels (c ^ 16) live on lane ^ 4
    y.x = __shfl_xor_sync(0xffffffffu, x.x, 4); y.y = __shfl_xor_sync(0xffffffffu, x.y, 4);
    y.z = __shfl_xor_sync(0xffffffffu, x.z, 4); y.w = __shfl_xor_sync(0xffffffffu, x.w, 4);
    float4 o;
    if (lo) {                                                        // a' = a cos - b sin
      o.x = x.x * cv.x - y.x * sv.x; o.y = x.y * cv.y - y.y * sv.y; o.z = x.z * cv.z - y.z * sv.z; o.w = x.w * cv.w - y.w * sv.w;
    } else {                                                         // b' = b cos + a sin
      o.x = x.x * cv.x + y.x * sv.x; o.y = x.y * cv.y + y.y * sv.y; o.z = x.z * cv.z + y.z * sv.z; o.w = x.w * cv.w + y.w * sv.w;
    }
    __nv_bfloat16* dst = t ? ko : qo;
    *reinterpret_cast<uint2*>(dst + (size_t)row * DM + 4 * lane) = make_uint2(pk(o.x, o.y), pk(o.z, o.w));
  }
}

// ------------------------------------------------------------------------------------------------ training (backward)
//   local_esm_qkrope_bwd  the backward of local_esm_qkrope, in place: dq, dk [M, 128] bf16 (rows ldd apart; the gradients of the rotated q / k)
//                         -> the gradients of the projections' q / k (pre-norm). Rotary transposed, (da', db') -> (da' cos + db' sin,
//                         db' cos - da' sin); RMSNorm backward per head from the saved q / k (recomputed statistics): u = x r, g = dy w,
//                         dx = r (g - u mean(g u)). The norm weights' gradients sum_rows dy u per channel (the 4 heads add) leave as per-block
//                         partials [gridDim.x, 64] (q 0..31, k 32..63), summed in block order by local_esm_qk_fin (deterministic).
//   local_esm_silu_bwd    ESMFold2's conditioning: every table is a product of sc = rn(silu(c)); g [M, 128] fp32 = the tables' gradient times their
//                         stacked weights -> dc = rn(g silu'(c)) and sc (the weight gradients' operand), bf16.
extern "C" __global__ void __launch_bounds__(256)
local_esm_qkrope_bwd(__nv_bfloat16* __restrict__ dq, __nv_bfloat16* __restrict__ dk, int ldd, const __nv_bfloat16* __restrict__ q,
                     const __nv_bfloat16* __restrict__ k, int ld, const float* __restrict__ wq, const float* __restrict__ wk,
                     const float* __restrict__ cs, const float* __restrict__ sn, float* __restrict__ part, int M, int n, float eps) {
  __shared__ float red[8][64];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int c = 4 * (lane & 7);
  const bool lo = c < 16;
  float gw[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
  for (int row = blockIdx.x * 8 + warp; row < M; row += gridDim.x * 8) {
    const int atom = row % n;
    const float4 cv = *reinterpret_cast<const float4*>(cs + (size_t)atom * 16 + (c & 15));
    const float4 sv = *reinterpret_cast<const float4*>(sn + (size_t)atom * 16 + (c & 15));
#pragma unroll
    for (int t = 0; t < 2; ++t) {
      __nv_bfloat16* g = (t ? dk : dq) + (size_t)row * ldd + 4 * lane;
      const float* wt = t ? wk : wq;
      const float4 x = ld4((t ? k : q) + (size_t)row * ld + 4 * lane);
      const float4 dy = ld4(g);
      float4 y;                                                      // the partner channels' gradient (c ^ 16 on lane ^ 4)
      y.x = __shfl_xor_sync(0xffffffffu, dy.x, 4); y.y = __shfl_xor_sync(0xffffffffu, dy.y, 4);
      y.z = __shfl_xor_sync(0xffffffffu, dy.z, 4); y.w = __shfl_xor_sync(0xffffffffu, dy.w, 4);
      float4 d;                                                      // the gradient before the rotary
      if (lo) {                                                      // a: da' cos + db' sin
        d.x = dy.x * cv.x + y.x * sv.x; d.y = dy.y * cv.y + y.y * sv.y; d.z = dy.z * cv.z + y.z * sv.z; d.w = dy.w * cv.w + y.w * sv.w;
      } else {                                                       // b: db' cos - da' sin
        d.x = dy.x * cv.x - y.x * sv.x; d.y = dy.y * cv.y - y.y * sv.y; d.z = dy.z * cv.z - y.z * sv.z; d.w = dy.w * cv.w - y.w * sv.w;
      }
      float ss = x.x * x.x + x.y * x.y + x.z * x.z + x.w * x.w;
      ss += __shfl_xor_sync(0xffffffffu, ss, 1); ss += __shfl_xor_sync(0xffffffffu, ss, 2); ss += __shfl_xor_sync(0xffffffffu, ss, 4);
      const float r = rsqrtf(ss * (1.f / 32.f) + eps);
      const float4 wv = *reinterpret_cast<const float4*>(wt + c);
      const float u0 = x.x * r, u1 = x.y * r, u2 = x.z * r, u3 = x.w * r;
      gw[t][0] += d.x * u0; gw[t][1] += d.y * u1; gw[t][2] += d.z * u2; gw[t][3] += d.w * u3;
      const float g0 = d.x * wv.x, g1 = d.y * wv.y, g2 = d.z * wv.z, g3 = d.w * wv.w;
      float m = g0 * u0 + g1 * u1 + g2 * u2 + g3 * u3;
      m += __shfl_xor_sync(0xffffffffu, m, 1); m += __shfl_xor_sync(0xffffffffu, m, 2); m += __shfl_xor_sync(0xffffffffu, m, 4);
      m *= 1.f / 32.f;
      *reinterpret_cast<uint2*>(g) = make_uint2(pk(r * (g0 - u0 * m), r * (g1 - u1 * m)), pk(r * (g2 - u2 * m), r * (g3 - u3 * m)));
    }
  }
  // the four heads of a lane octet add (lanes l, l ^ 8, l ^ 16, l ^ 24), then the warps in order
#pragma unroll
  for (int t = 0; t < 2; ++t)
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      float v = gw[t][e];
      v += __shfl_xor_sync(0xffffffffu, v, 8); v += __shfl_xor_sync(0xffffffffu, v, 16);
      if (lane < 8) red[warp][32 * t + c + e] = v;
    }
  __syncthreads();
  if (threadIdx.x < 64) {
    float s = 0.f;
#pragma unroll
    for (int w = 0; w < 8; ++w) s += red[w][threadIdx.x];
    part[(size_t)blockIdx.x * 64 + threadIdx.x] = s;
  }
}

// part [nb, 64] -> out [64] (q's weight 0..31, k's 32..63): 512 threads, thread (j, e) sums the blocks j, j + 8, .. of entry e, then the eight
// sums add in order (a fixed order: deterministic)
extern "C" __global__ void __launch_bounds__(512)
local_esm_qk_fin(const float* __restrict__ part, int nb, float* __restrict__ out) {
  __shared__ float red[8][64];
  const int e = threadIdx.x & 63, j = threadIdx.x >> 6;
  float s = 0.f;
  for (int b = j; b < nb; b += 8) s += part[(size_t)b * 64 + e];
  red[j][e] = s;
  __syncthreads();
  if (threadIdx.x < 64) {
    float t = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) t += red[i][e];
    out[e] = t;
  }
}

extern "C" __global__ void __launch_bounds__(256)
local_esm_silu_bwd(const __nv_bfloat16* __restrict__ c, const float* __restrict__ g, __nv_bfloat16* __restrict__ dc, __nv_bfloat16* __restrict__ sc,
                   long long n4) {
  for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += (long long)gridDim.x * blockDim.x) {
    const float4 x = ld4(c + 4 * i);
    const float4 gv = *reinterpret_cast<const float4*>(g + 4 * i);
    const float xs[4] = {x.x, x.y, x.z, x.w}, gs[4] = {gv.x, gv.y, gv.z, gv.w};
    float d[4], s[4];
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const float sg = 1.f / (1.f + __expf(-xs[e]));
      s[e] = xs[e] * sg;
      d[e] = gs[e] * sg * (1.f + xs[e] * (1.f - sg));
    }
    *reinterpret_cast<uint2*>(dc + 4 * i) = make_uint2(pk(d[0], d[1]), pk(d[2], d[3]));
    *reinterpret_cast<uint2*>(sc + 4 * i) = make_uint2(pk(s[0], s[1]), pk(s[2], s[3]));
  }
}
