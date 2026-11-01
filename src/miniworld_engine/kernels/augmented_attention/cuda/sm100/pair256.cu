// pair256.cu — AugmentedAttentionPairBias's pair -> bias pass for an fp32 256-wide pair at 16 heads (ESMFold2's token layout,
// d_single 768 / 16 x 48, d_pair 256) on the module route's fp32 (TF32 core) path (``module_route.py``), forward and backward, fp32
// CUDA cores (the route's other fp32 pair pass, bias_only_dit's rows32, is d_pair 128).
//
//   aap256_fwd  bias[h, i, j] = rstd (wf[h] . p - mean sum_k wf[h, k]) for the pair row p = pair[i, j, :] (two-pass LayerNorm
//               statistics), wf = wb diag(lnw) (exp2 units when the caller scaled lnw), NEG on masked keys (see the kernel).
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

// forward: a warp takes tiles of FR = 16 consecutive pair rows (16 KB), staged in shared memory by cp.async two tiles deep. The row
// statistics come from the tile (two passes, a half row a lane), then the 16 x 16 head projection of the CENTERED rows runs as TF32
// mma.sync m16n8k8 (32 k-steps x 2 head tiles; operands rounded to nearest TF32, fp32 accumulation -- the TF32 grid of the PyTorch
// statements' Linear over the normalized pair), scaled by rstd in the epilogue. (Earlier forms on fp32 CUDA cores ran 494 / 288 us
// at L768: shuffle reduce-scatter per row, then shared-memory weights re-read per FMA.)
constexpr int FR = 16, FW = 6, RS = DP + 4, WS = 24;   // rows a tile, warps a CTA, padded tile row stride, padded weight row stride

__device__ __forceinline__ void cp16(float* dst, const float* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(static_cast<uint32_t>(__cvta_generic_to_shared(dst))), "l"(src)
               : "memory");
}
__device__ __forceinline__ uint32_t tf32(float x) {
  uint32_t u;
  asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(u) : "f"(x));
  return u;
}
__device__ __forceinline__ void mma_tf32(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

extern "C" __global__ void __launch_bounds__(FW * 32, 1) aap256_fwd(const float* __restrict__ pair, const float* __restrict__ wf,
                                                                     const bool* __restrict__ m, float* __restrict__ bias, int Lb, float eps,
                                                                     float neg) {
  extern __shared__ __align__(16) float sm[];
  uint32_t* wt = reinterpret_cast<uint32_t*>(sm);   // [DP][WS] TF32 bits, heads 0..15 of row k (B[k][n])
  float* tiles = sm + DP * WS;                       // [FW][2][FR][RS]
  for (int i = threadIdx.x; i < NH * DP; i += blockDim.x) {
    const int hh = i / DP, k = i % DP;
    wt[k * WS + hh] = tf32(wf[i]);
  }
  __syncthreads();
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int g = lane >> 2, t = lane & 3, sr = lane & 15, half = lane >> 4;
  float* tw = tiles + warp * 2 * FR * RS;
  const int64_t plane = int64_t(Lb) * Lb, groups = plane / FR;
  const int64_t stride = int64_t(gridDim.x) * FW;
  auto stage = [&](int buf, int64_t gi) {
    const float* src = pair + gi * FR * DP;
    float* dst = tw + buf * FR * RS;
#pragma unroll
    for (int i = 0; i < FR * DP / 4 / 32; ++i) {
      const int c = i * 32 + lane, row = c / (DP / 4), col = (c % (DP / 4)) * 4;
      cp16(dst + row * RS + col, src + row * DP + col);
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
  };
  int64_t gi = int64_t(blockIdx.x) * FW + warp;
  if (gi < groups) stage(0, gi);
  for (int it = 0; gi < groups; gi += stride, ++it) {
    const int buf = it & 1;
    if (gi + stride < groups) {
      stage(buf ^ 1, gi + stride);
      asm volatile("cp.async.wait_group 1;" ::: "memory");
    } else {
      asm volatile("cp.async.wait_group 0;" ::: "memory");
    }
    __syncwarp();
    const float* x = tw + buf * FR * RS;
    // statistics: lane (sr, half) sums half a row; the halves meet by one shuffle
    const float* xs = x + sr * RS + half * (DP / 2);
    float s = 0.f;
#pragma unroll 8
    for (int k = 0; k < DP / 2; k += 4) {
      const float4 v = *reinterpret_cast<const float4*>(xs + k);
      s += (v.x + v.y) + (v.z + v.w);
    }
    s += __shfl_xor_sync(0xffffffffu, s, 16);
    const float mean = s * (1.f / DP);
    float q = 0.f;
#pragma unroll 8
    for (int k = 0; k < DP / 2; k += 4) {
      const float4 v = *reinterpret_cast<const float4*>(xs + k);
      q += (v.x - mean) * (v.x - mean) + (v.y - mean) * (v.y - mean) + (v.z - mean) * (v.z - mean) + (v.w - mean) * (v.w - mean);
    }
    q += __shfl_xor_sync(0xffffffffu, q, 16);
    const float rstd = rsqrtf(q * (1.f / DP) + eps);
    const float m0 = __shfl_sync(0xffffffffu, mean, g), m1 = __shfl_sync(0xffffffffu, mean, g + 8);
    const float r0 = __shfl_sync(0xffffffffu, rstd, g), r1 = __shfl_sync(0xffffffffu, rstd, g + 8);
    float c[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
    const float* xa = x + g * RS + t;
    const float* xb = x + (g + 8) * RS + t;
#pragma unroll 8
    for (int k0 = 0; k0 < DP; k0 += 8) {
      const uint32_t a0 = tf32(xa[k0] - m0), a1 = tf32(xb[k0] - m1), a2 = tf32(xa[k0 + 4] - m0), a3 = tf32(xb[k0 + 4] - m1);
#pragma unroll
      for (int nt = 0; nt < 2; ++nt)
        mma_tf32(c[nt], a0, a1, a2, a3, wt[(k0 + t) * WS + nt * 8 + g], wt[(k0 + t + 4) * WS + nt * 8 + g]);
    }
    const int64_t row0 = gi * FR + g, row1 = row0 + 8;
    const bool l0 = m == nullptr || m[int(row0 % Lb)], l1 = m == nullptr || m[int(row1 % Lb)];
#pragma unroll
    for (int nt = 0; nt < 2; ++nt) {
      const int hh = nt * 8 + 2 * t;
      bias[hh * plane + row0] = l0 ? r0 * c[nt][0] : neg;
      bias[(hh + 1) * plane + row0] = l0 ? r0 * c[nt][1] : neg;
      bias[hh * plane + row1] = l1 ? r1 * c[nt][2] : neg;
      bias[(hh + 1) * plane + row1] = l1 ? r1 * c[nt][3] : neg;
    }
    __syncwarp();   // the buffer is refilled by the next iteration's stage()
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
