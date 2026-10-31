// pair16.cu — AugmentedAttentionPairBias's pair -> bias pass at a 16-wide pair (the atom layout: d_single 128, 4 heads x 32,
// d_pair 16; ``module_route.py``), forward and backward, plain CUDA cores (a 16-wide row is 32 / 64 bytes: no tile for an MMA).
//
//   aap16_fwd_*   bias[h, i, j] = rstd (wf[h] . p - mean sum_k wf[h, k]) with (mean, rstd) the LayerNorm statistics of the pair row
//                 p = pair[i, j, :] (two passes over its 16 values) and wf = wb diag(lnw) (the LayerNorm weight folded into the head
//                 projection, exp2 units when the caller scaled it); a masked key j (m[j] false) gets NEG. One thread: J consecutive
//                 keys of one query row (J 16-byte stores per head plane). Suffix: pair dtype, then bias dtype (b = bf16, f = fp32).
//   aap16_bwd_*   from dbias DB [H, Lb, Lb] fp32: dpair = rstd (dxh - mean(dxh) - xh mean(dxh xh)) with dxh = sum_h DB[h] wf[h] and
//                 xh the normalized row, and per-block partials of dwf[h, k] = sum_rows DB[h] xh[k] (fixed order: per thread, the
//                 warp's lanes by shuffles, then the block's warps in order; the caller sums the block rows in order -- bitwise
//                 repeatable). Suffix: pair dtype.
//
// Rows: Lb a multiple of 16 (the pass's padded length), so a thread's J keys never straddle a query row.
// SPDX-License-Identifier: Apache-2.0
#include <cuda_bf16.h>
#include <cstdint>

typedef __nv_bfloat16 bf16;
constexpr int DP = 16;   // pair width
constexpr int NH = 4;    // heads

template <typename T> struct Row16;
template <> struct Row16<bf16> { uint4 u[2]; };      // 16 bf16
template <> struct Row16<float> { float4 u[4]; };    // 16 fp32

__device__ __forceinline__ void unpack(const Row16<bf16>& r, float* v) {
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    const uint32_t w[4] = {r.u[i].x, r.u[i].y, r.u[i].z, r.u[i].w};
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const float2 f = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&w[k]));
      v[i * 8 + 2 * k] = f.x;
      v[i * 8 + 2 * k + 1] = f.y;
    }
  }
}
__device__ __forceinline__ void unpack(const Row16<float>& r, float* v) {
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    v[4 * i] = r.u[i].x; v[4 * i + 1] = r.u[i].y; v[4 * i + 2] = r.u[i].z; v[4 * i + 3] = r.u[i].w;
  }
}
template <typename T> __device__ __forceinline__ Row16<T> ld_row(const T* p) { return *reinterpret_cast<const Row16<T>*>(p); }

__device__ __forceinline__ void st_row(bf16* p, const float* v) {
  uint4 o[2];
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    uint32_t w[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      __nv_bfloat162 h = __floats2bfloat162_rn(v[i * 8 + 2 * k], v[i * 8 + 2 * k + 1]);
      w[k] = *reinterpret_cast<uint32_t*>(&h);
    }
    o[i] = make_uint4(w[0], w[1], w[2], w[3]);
  }
  reinterpret_cast<uint4*>(p)[0] = o[0];
  reinterpret_cast<uint4*>(p)[1] = o[1];
}
__device__ __forceinline__ void st_row(float* p, const float* v) {
#pragma unroll
  for (int i = 0; i < 4; ++i) reinterpret_cast<float4*>(p)[i] = make_float4(v[4 * i], v[4 * i + 1], v[4 * i + 2], v[4 * i + 3]);
}

// J consecutive values of one head plane
template <int J> __device__ __forceinline__ void st_keys(bf16* p, const float* v) {
  static_assert(J == 4 || J == 8, "J");
  uint32_t w[J / 2];
#pragma unroll
  for (int k = 0; k < J / 2; ++k) {
    __nv_bfloat162 h = __floats2bfloat162_rn(v[2 * k], v[2 * k + 1]);
    w[k] = *reinterpret_cast<uint32_t*>(&h);
  }
  if constexpr (J == 8) *reinterpret_cast<uint4*>(p) = make_uint4(w[0], w[1], w[2], w[3]);
  else *reinterpret_cast<uint2*>(p) = make_uint2(w[0], w[1]);
}
template <int J> __device__ __forceinline__ void st_keys(float* p, const float* v) {
#pragma unroll
  for (int k = 0; k < J; k += 4) *reinterpret_cast<float4*>(p + k) = make_float4(v[k], v[k + 1], v[k + 2], v[k + 3]);
}

// the LayerNorm statistics of a 16-value row (two passes)
__device__ __forceinline__ void stats16(const float* v, float eps, float& mean, float& rstd) {
  float s = 0.f;
#pragma unroll
  for (int k = 0; k < DP; ++k) s += v[k];
  mean = s * (1.f / DP);
  float q = 0.f;
#pragma unroll
  for (int k = 0; k < DP; ++k) {
    const float d = v[k] - mean;
    q += d * d;
  }
  rstd = rsqrtf(q * (1.f / DP) + eps);
}

template <typename P, typename O, int J>
__device__ __forceinline__ void fwd_body(const P* __restrict__ pair, const float* __restrict__ wf, const bool* __restrict__ m, O* __restrict__ bias,
                                         int Lb, float eps, float neg) {
  __shared__ float4 w4[NH * DP / 4];   // wf [NH][16]
  __shared__ float ws[NH];             // sum_k wf[h, k]
  if (threadIdx.x < NH * DP / 4) w4[threadIdx.x] = reinterpret_cast<const float4*>(wf)[threadIdx.x];
  if (threadIdx.x < NH) {
    float s = 0.f;
    for (int k = 0; k < DP; ++k) s += wf[threadIdx.x * DP + k];
    ws[threadIdx.x] = s;
  }
  __syncthreads();
  const int64_t groups = int64_t(Lb) * Lb / J;
  const int64_t plane = int64_t(Lb) * Lb;
  for (int64_t gi = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; gi < groups; gi += int64_t(gridDim.x) * blockDim.x) {
    const int64_t r0 = gi * J;                 // first pair row (= i Lb + j0)
    const int j0 = int(r0 % Lb);
    Row16<P> raw[J];
#pragma unroll
    for (int t = 0; t < J; ++t) raw[t] = ld_row<P>(pair + (r0 + t) * DP);
    float out[NH][J];
#pragma unroll
    for (int t = 0; t < J; ++t) {
      float v[DP];
      unpack(raw[t], v);
      float mean, rstd;
      stats16(v, eps, mean, rstd);
      const bool live = m == nullptr || m[j0 + t];
#pragma unroll
      for (int h = 0; h < NH; ++h) {
        float d = 0.f;
#pragma unroll
        for (int k4 = 0; k4 < DP / 4; ++k4) {
          const float4 w = w4[h * (DP / 4) + k4];
          d += w.x * v[4 * k4] + w.y * v[4 * k4 + 1] + w.z * v[4 * k4 + 2] + w.w * v[4 * k4 + 3];
        }
        out[h][t] = live ? rstd * (d - mean * ws[h]) : neg;
      }
    }
#pragma unroll
    for (int h = 0; h < NH; ++h) st_keys<J>(bias + h * plane + r0, out[h]);
  }
}

extern "C" __global__ void __launch_bounds__(256) aap16_fwd_bb(const bf16* pair, const float* wf, const bool* m, bf16* bias, int Lb, float eps, float neg) {
  fwd_body<bf16, bf16, 8>(pair, wf, m, bias, Lb, eps, neg);
}
extern "C" __global__ void __launch_bounds__(256) aap16_fwd_bf(const bf16* pair, const float* wf, const bool* m, float* bias, int Lb, float eps, float neg) {
  fwd_body<bf16, float, 8>(pair, wf, m, bias, Lb, eps, neg);
}
extern "C" __global__ void __launch_bounds__(256) aap16_fwd_fb(const float* pair, const float* wf, const bool* m, bf16* bias, int Lb, float eps, float neg) {
  fwd_body<float, bf16, 4>(pair, wf, m, bias, Lb, eps, neg);
}
extern "C" __global__ void __launch_bounds__(256) aap16_fwd_ff(const float* pair, const float* wf, const bool* m, float* bias, int Lb, float eps, float neg) {
  fwd_body<float, float, 4>(pair, wf, m, bias, Lb, eps, neg);
}

// backward: J = 4 consecutive rows per thread (one float4 of each dbias plane)
template <typename P>
__device__ __forceinline__ void bwd_body(const float* __restrict__ DB, const P* __restrict__ pair, const float* __restrict__ wf, P* __restrict__ dpair,
                                         float* __restrict__ part, int Lb, float eps) {
  constexpr int J = 4;
  __shared__ float4 w4[NH * DP / 4];
  __shared__ float red[8][NH * DP];
  if (threadIdx.x < NH * DP / 4) w4[threadIdx.x] = reinterpret_cast<const float4*>(wf)[threadIdx.x];
  __syncthreads();
  float acc[NH][DP];
#pragma unroll
  for (int h = 0; h < NH; ++h)
#pragma unroll
    for (int k = 0; k < DP; ++k) acc[h][k] = 0.f;
  const int64_t plane = int64_t(Lb) * Lb;
  const int64_t groups = plane / J;
  for (int64_t gi = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; gi < groups; gi += int64_t(gridDim.x) * blockDim.x) {
    const int64_t r0 = gi * J;
    float4 g4[NH];
#pragma unroll
    for (int h = 0; h < NH; ++h) g4[h] = *reinterpret_cast<const float4*>(DB + h * plane + r0);
    Row16<P> raw[J];
#pragma unroll
    for (int t = 0; t < J; ++t) raw[t] = ld_row<P>(pair + (r0 + t) * DP);
#pragma unroll
    for (int t = 0; t < J; ++t) {
      float v[DP];
      unpack(raw[t], v);
      float mean, rstd;
      stats16(v, eps, mean, rstd);
      float g[NH];
#pragma unroll
      for (int h = 0; h < NH; ++h) g[h] = t == 0 ? g4[h].x : t == 1 ? g4[h].y : t == 2 ? g4[h].z : g4[h].w;
      float dxh[DP];
      float s1 = 0.f, s2 = 0.f;
#pragma unroll
      for (int k4 = 0; k4 < DP / 4; ++k4) {
        float4 d = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
        for (int h = 0; h < NH; ++h) {
          const float4 w = w4[h * (DP / 4) + k4];
          d.x += g[h] * w.x; d.y += g[h] * w.y; d.z += g[h] * w.z; d.w += g[h] * w.w;
        }
        dxh[4 * k4] = d.x; dxh[4 * k4 + 1] = d.y; dxh[4 * k4 + 2] = d.z; dxh[4 * k4 + 3] = d.w;
      }
#pragma unroll
      for (int k = 0; k < DP; ++k) {
        const float xh = (v[k] - mean) * rstd;
        v[k] = xh;
        s1 += dxh[k];
        s2 += dxh[k] * xh;
#pragma unroll
        for (int h = 0; h < NH; ++h) acc[h][k] += g[h] * xh;
      }
      s1 *= 1.f / DP;
      s2 *= 1.f / DP;
#pragma unroll
      for (int k = 0; k < DP; ++k) dxh[k] = rstd * (dxh[k] - s1 - v[k] * s2);
      st_row(dpair + (r0 + t) * DP, dxh);
    }
  }
  // the block's dwf partial: lanes by shuffles (fixed tree), then the warps in order
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
#pragma unroll
  for (int h = 0; h < NH; ++h)
#pragma unroll
    for (int k = 0; k < DP; ++k) {
      float s = acc[h][k];
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
      if (lane == 0) red[warp][h * DP + k] = s;
    }
  __syncthreads();
  if (threadIdx.x < NH * DP) {
    float s = 0.f;
    for (int w = 0; w < (int)(blockDim.x >> 5); ++w) s += red[w][threadIdx.x];
    part[int64_t(blockIdx.x) * NH * DP + threadIdx.x] = s;
  }
}

extern "C" __global__ void __launch_bounds__(256) aap16_bwd_b(const float* DB, const bf16* pair, const float* wf, bf16* dpair, float* part, int Lb, float eps) {
  bwd_body<bf16>(DB, pair, wf, dpair, part, Lb, eps);
}
extern "C" __global__ void __launch_bounds__(256) aap16_bwd_f(const float* DB, const float* pair, const float* wf, float* dpair, float* part, int Lb, float eps) {
  bwd_body<float>(DB, pair, wf, dpair, part, Lb, eps);
}
