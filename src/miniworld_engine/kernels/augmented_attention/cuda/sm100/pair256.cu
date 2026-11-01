// pair256.cu — AugmentedAttentionPairBias's pair -> bias pass for an fp32 256-wide pair at 16 heads (ESMFold2's token layout,
// d_single 768 / 16 x 48, d_pair 256) on the module route's fp32 (TF32 core) path (``module_route.py``), forward and backward, fp32
// CUDA cores (the route's other fp32 pair pass, bias_only_dit's rows32, is d_pair 128).
//
//   aap256_fwd  bias[h, i, j] = rstd (wf[h] . p - mean sum_k wf[h, k]) for the pair row p = pair[i, j, :] (two-pass LayerNorm
//               statistics), wf = wb diag(lnw) (exp2 units when the caller scaled lnw), NEG on masked keys. A warp takes 16
//               consecutive rows (one query row's keys j0..j0+15): a lane holds 8 features of a row, the 16 head dot products are
//               reduce-scattered over the warp (16 shuffles: lane l ends with head l >> 1), and each even lane stores its head's 16
//               keys as four float4.
//   aap256_bwd  from dbias DB [16, Lb, Lb] fp32: dpair = rstd (dxh - mean(dxh) - xh mean(dxh xh)), dxh = sum_h DB[h] wf[h], and
//               per-CTA partials of dwf[h, k] = sum_rows DB[h] xh[k] (each lane owns 8 features x 16 heads; the CTA's warps added
//               in order in shared memory): fixed order, the caller sums the CTA rows in order.
// Rows: Lb a multiple of 16.
// SPDX-License-Identifier: Apache-2.0
#include <cstdint>

constexpr int DP = 256, NH = 16, G = 16, NW = 8;

__device__ __forceinline__ void ld8(const float* p, float* v) {
  const float4 a = reinterpret_cast<const float4*>(p)[0], b = reinterpret_cast<const float4*>(p)[1];
  v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w; v[4] = b.x; v[5] = b.y; v[6] = b.z; v[7] = b.w;
}

__device__ __forceinline__ float wsum(float s) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
  return s;
}

// 16 partials a lane -> the warp-wide sum of head (lane >> 1) on every lane (both lanes of a pair hold it)
__device__ __forceinline__ float reduce_scatter16(float* d, int lane) {
#pragma unroll
  for (int step = 0; step < 4; ++step) {
    const int o = 16 >> step, c = 8 >> step;
    const bool hi = lane & o;
#pragma unroll
    for (int i = 0; i < c; ++i) {
      const float send = hi ? d[i] : d[c + i];
      const float keep = hi ? d[c + i] : d[i];
      d[i] = keep + __shfl_xor_sync(0xffffffffu, send, o);
    }
  }
  return d[0] + __shfl_xor_sync(0xffffffffu, d[0], 1);
}

extern "C" __global__ void __launch_bounds__(NW * 32) aap256_fwd(const float* __restrict__ pair, const float* __restrict__ wf,
                                                                  const bool* __restrict__ m, float* __restrict__ bias, int Lb, float eps,
                                                                  float neg) {
  __shared__ __align__(16) float w[NH * DP];
  __shared__ float ws[NH];
  for (int i = threadIdx.x; i < NH * DP / 4; i += blockDim.x) reinterpret_cast<float4*>(w)[i] = reinterpret_cast<const float4*>(wf)[i];
  __syncthreads();
  if (threadIdx.x < NH) {
    float s = 0.f;
    for (int k = 0; k < DP; ++k) s += w[threadIdx.x * DP + k];
    ws[threadIdx.x] = s;
  }
  __syncthreads();
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int h = lane >> 1;
  const int64_t plane = int64_t(Lb) * Lb, groups = plane / G;
  for (int64_t gi = int64_t(blockIdx.x) * NW + warp; gi < groups; gi += int64_t(gridDim.x) * NW) {
    const int64_t r0 = gi * G;
    const int j0 = int(r0 % Lb);
    float out[G];
#pragma unroll
    for (int t = 0; t < G; ++t) {
      float v[8];
      ld8(pair + (r0 + t) * DP + lane * 8, v);
      float s = 0.f;
#pragma unroll
      for (int k = 0; k < 8; ++k) s += v[k];
      const float mean = wsum(s) * (1.f / DP);
      float q = 0.f;
#pragma unroll
      for (int k = 0; k < 8; ++k) q += (v[k] - mean) * (v[k] - mean);
      const float rstd = rsqrtf(wsum(q) * (1.f / DP) + eps);
      float d[NH];
#pragma unroll
      for (int hh = 0; hh < NH; ++hh) {
        const float4 a = reinterpret_cast<const float4*>(w + hh * DP + lane * 8)[0], b = reinterpret_cast<const float4*>(w + hh * DP + lane * 8)[1];
        d[hh] = a.x * v[0] + a.y * v[1] + a.z * v[2] + a.w * v[3] + b.x * v[4] + b.y * v[5] + b.z * v[6] + b.w * v[7];
      }
      const float dot = reduce_scatter16(d, lane);
      out[t] = (m == nullptr || m[j0 + t]) ? rstd * (dot - mean * ws[h]) : neg;
    }
    if ((lane & 1) == 0) {
      float4* o = reinterpret_cast<float4*>(bias + h * plane + r0);
#pragma unroll
      for (int i = 0; i < G / 4; ++i) o[i] = make_float4(out[4 * i], out[4 * i + 1], out[4 * i + 2], out[4 * i + 3]);
    }
  }
}

extern "C" __global__ void __launch_bounds__(NW * 32, 1) aap256_bwd(const float* __restrict__ DB, const float* __restrict__ pair,
                                                                     const float* __restrict__ wf, float* __restrict__ dpair,
                                                                     float* __restrict__ part, int Lb, float eps) {
  __shared__ __align__(16) float w[NH * DP];
  __shared__ __align__(16) float acc_s[NH * DP];
  for (int i = threadIdx.x; i < NH * DP / 4; i += blockDim.x) {
    reinterpret_cast<float4*>(w)[i] = reinterpret_cast<const float4*>(wf)[i];
    reinterpret_cast<float4*>(acc_s)[i] = make_float4(0.f, 0.f, 0.f, 0.f);
  }
  __syncthreads();
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int hl = lane & 15;   // the head whose dbias this lane loads
  const int64_t plane = int64_t(Lb) * Lb, groups = plane / G;
  float acc[NH][8];
#pragma unroll
  for (int hh = 0; hh < NH; ++hh)
#pragma unroll
    for (int k = 0; k < 8; ++k) acc[hh][k] = 0.f;
  for (int64_t gi = int64_t(blockIdx.x) * NW + warp; gi < groups; gi += int64_t(gridDim.x) * NW) {
    const int64_t r0 = gi * G;
    float gv[G];
    {
      const float4* src = reinterpret_cast<const float4*>(DB + hl * plane + r0);
#pragma unroll
      for (int i = 0; i < G / 4; ++i) {
        const float4 a = src[i];
        gv[4 * i] = a.x; gv[4 * i + 1] = a.y; gv[4 * i + 2] = a.z; gv[4 * i + 3] = a.w;
      }
    }
#pragma unroll
    for (int t = 0; t < G; ++t) {
      float v[8];
      ld8(pair + (r0 + t) * DP + lane * 8, v);
      float s = 0.f;
#pragma unroll
      for (int k = 0; k < 8; ++k) s += v[k];
      const float mean = wsum(s) * (1.f / DP);
      float q = 0.f;
#pragma unroll
      for (int k = 0; k < 8; ++k) q += (v[k] - mean) * (v[k] - mean);
      const float rstd = rsqrtf(wsum(q) * (1.f / DP) + eps);
#pragma unroll
      for (int k = 0; k < 8; ++k) v[k] = (v[k] - mean) * rstd;   // xh
      float dxh[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int hh = 0; hh < NH; ++hh) {
        const float g = __shfl_sync(0xffffffffu, gv[t], hh);
        const float4 a = reinterpret_cast<const float4*>(w + hh * DP + lane * 8)[0], b = reinterpret_cast<const float4*>(w + hh * DP + lane * 8)[1];
        dxh[0] += g * a.x; dxh[1] += g * a.y; dxh[2] += g * a.z; dxh[3] += g * a.w;
        dxh[4] += g * b.x; dxh[5] += g * b.y; dxh[6] += g * b.z; dxh[7] += g * b.w;
#pragma unroll
        for (int k = 0; k < 8; ++k) acc[hh][k] += g * v[k];
      }
      float s1 = 0.f, s2 = 0.f;
#pragma unroll
      for (int k = 0; k < 8; ++k) {
        s1 += dxh[k];
        s2 += dxh[k] * v[k];
      }
      s1 = wsum(s1) * (1.f / DP);
      s2 = wsum(s2) * (1.f / DP);
      float4* o = reinterpret_cast<float4*>(dpair + (r0 + t) * DP + lane * 8);
      o[0] = make_float4(rstd * (dxh[0] - s1 - v[0] * s2), rstd * (dxh[1] - s1 - v[1] * s2), rstd * (dxh[2] - s1 - v[2] * s2),
                         rstd * (dxh[3] - s1 - v[3] * s2));
      o[1] = make_float4(rstd * (dxh[4] - s1 - v[4] * s2), rstd * (dxh[5] - s1 - v[5] * s2), rstd * (dxh[6] - s1 - v[6] * s2),
                         rstd * (dxh[7] - s1 - v[7] * s2));
    }
  }
  // the CTA's warps in order (each lane owns 8 features x 16 heads: no overlap inside a warp)
  for (int wi = 0; wi < NW; ++wi) {
    if (warp == wi) {
#pragma unroll
      for (int hh = 0; hh < NH; ++hh)
#pragma unroll
        for (int k = 0; k < 8; ++k) acc_s[hh * DP + lane * 8 + k] += acc[hh][k];
    }
    __syncthreads();
  }
  for (int i = threadIdx.x; i < NH * DP; i += blockDim.x) part[int64_t(blockIdx.x) * NH * DP + i] = acc_s[i];
}
