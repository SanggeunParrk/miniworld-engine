// embed_glue.cu — the InputFeatureEmbedder's own statements around its SWA atom transformer on sm_100a (B200 2026-11-17, round 3), each a
// single launch instead of a chain of small torch launches (one system of a few thousand atoms: the glue was ~50 of the step's ~65 launches):
//   swa_embed_rope_sm100     the 3D RoPE tables cos / sin [M, half] fp32 (functional.build_3d_rope: angle j = source_j * inv_freq_j, source the
//                            x / y / z coordinate, the space uid, or nothing (cos 1, sin 0)) and seqused [B] = valid atoms per row
//                            (build_attention_params), the count by B extra blocks (one per row; integer sums, no atomics)
//   swa_embed_feat_sm100     inference: the reference features, their bias-free projection W [128, n_in] (read as W^T rows) and, for the
//                            "esmfold2" layout, the affine LayerNorm over the 128 channels, in fp32; out bf16 or fp32. One warp per atom.
//                              legacy    f = [x, y, z, m, element, asinh(charge)] m            (m: ref_mask)
//                              esmfold2  f = [x, y, z, charge, m, onehot_128(element)] m       (m: atom_mask)
//   swa_embed_scatter_sm100  inference read-out: acc[b, token, :T] += sum of relu?(x[atom, :]) over a run of valid atoms of one token, acc[b, token, T] +=
//                            their count (fp32 red.add into a zeroed [B, n_token, T + 1]: one add per cell when a token's atoms are contiguous)
//   swa_embed_tokmean_sm100  out = acc[..., :T] / max(acc[..., T], 1)   (bf16 or fp32)
// No programmatic dependent launch.
// SPDX-License-Identifier: Apache-2.0
#include <stdint.h>

constexpr int C = 128;

__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) {
  uint32_t r;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo));
  return r;
}
__device__ __forceinline__ float bf16lo(uint32_t v) { return __uint_as_float(v << 16); }
__device__ __forceinline__ float bf16hi(uint32_t v) { return __uint_as_float(v & 0xffff0000u); }
// round 3 part 14 (flags): bit 0 -- the positions are bf16 (read in place, widened: the values of their fp32 cast); bit 1 -- the projection
// weight is read as W [128, nf] (the parameter itself) instead of W^T [nf, 128]; featbwd bit 2 -- DPOS in bf16 (the positions' dtype)
__device__ __forceinline__ float ldpos(const void* P, long long i, int fl) {
  return (fl & 1) ? __uint_as_float((uint32_t)reinterpret_cast<const uint16_t*>(P)[i] << 16) : reinterpret_cast<const float*>(P)[i];
}
__device__ __forceinline__ float4 ldw(const float* W, int k, int lane, int nf, int fl) {
  if (fl & 2) {
    const float* p = W + (size_t)(4 * lane) * nf + k;
    return make_float4(p[0], p[nf], p[2 * nf], p[3 * nf]);
  }
  return *reinterpret_cast<const float4*>(W + (size_t)k * C + 4 * lane);
}
// N1 round 4: nonzero bytes of a word (a byte is 0 or nonzero); the seqused count of a row on 16-byte loads (VEC). The count block of a row is the kernel tail
// (256 threads, N / 256 one-byte rounds: ~8 - 11 us at 4096 - 6144 atoms; the rope blocks are done long before)
__device__ __forceinline__ int nz4(uint32_t w) { return __popc((((w & 0x7f7f7f7fu) + 0x7f7f7f7fu) | w) & 0x80808080u); }
template <bool VEC>
__device__ __forceinline__ void rope_body(const void* __restrict__ POS, const long long* __restrict__ UID, const uint8_t* __restrict__ MASK,
                     const float* __restrict__ INV, const int* __restrict__ SRC, float* __restrict__ COS, float* __restrict__ SIN,
                     int* __restrict__ SEQ, int B, int N, int half, int nrope, int fl) {
  if ((int)blockIdx.x >= nrope) {                                         // seqused of row b
    const int b = (int)blockIdx.x - nrope;
    const uint8_t* m = MASK + (size_t)b * N;
    int cnt = 0;
    if (VEC && (N & 15) == 0) {
      const uint4* m4 = reinterpret_cast<const uint4*>(m);
      for (int i = threadIdx.x; i < (N >> 4); i += blockDim.x) {
        const uint4 u = m4[i];
        cnt += nz4(u.x) + nz4(u.y) + nz4(u.z) + nz4(u.w);
      }
    } else {
      for (int i = threadIdx.x; i < N; i += blockDim.x) cnt += m[i] != 0;
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) cnt += __shfl_xor_sync(0xffffffffu, cnt, o);
    __shared__ int part[8];
    if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = cnt;
    __syncthreads();
    if (threadIdx.x == 0) {
      int s = 0;
      for (int w = 0; w < (int)blockDim.x / 32; ++w) s += part[w];
      SEQ[b] = s;
    }
    return;
  }
  const long long total = (long long)B * N * half;
  const long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= total) return;
  const long long m = t / half;
  const int j = (int)(t - m * half), src = SRC[j];
  float a = 0.f;
  if (src < 3) a = ldpos(POS, m * 3 + src, fl) * INV[j];
  else if (src == 3) a = (float)UID[m] * INV[j];
  float s, c;
  sincosf(a, &s, &c);
  COS[t] = c;
  SIN[t] = s;
}

// one warp per atom, lane = channels 4 lane .. 4 lane + 3
extern "C" __global__ void __launch_bounds__(256)
swa_embed_feat_sm100(const void* __restrict__ POS, const long long* __restrict__ CHG, const long long* __restrict__ ELEM,
                     const uint8_t* __restrict__ MASK, const float* __restrict__ WT, const float* __restrict__ LNW,
                     const float* __restrict__ LNB, void* __restrict__ OUT, int M, int mode, int out_bf16, int ln, float eps,
                     float* __restrict__ PROJ, int fl) {   // PROJ (training, round 3 part 11): the masked projection before the LayerNorm, fp32 [M, 128] (or null)
  const int lane = threadIdx.x & 31;
  const int nf = mode == 0 ? 6 : 133;
  for (long long r = (long long)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5); r < M; r += (long long)gridDim.x * (blockDim.x >> 5)) {
    const float m = MASK[r] ? 1.f : 0.f;
    const float x = ldpos(POS, r * 3, fl), y = ldpos(POS, r * 3 + 1, fl), z = ldpos(POS, r * 3 + 2, fl);
    auto w = [&](int k) { return ldw(WT, k, lane, nf, fl); };
    float f3, f4, f5;
    float4 w5;
    if (mode == 0) {                                                      // legacy: [x, y, z, m, element, asinh(charge)]
      f3 = m, f4 = (float)ELEM[r], f5 = asinhf((float)CHG[r]);
      w5 = w(5);
    } else {                                                              // esmfold2: [x, y, z, charge, m, onehot(element)]
      long long e = ELEM[r];
      e = e < 0 ? 0 : (e > 127 ? 127 : e);
      f3 = (float)CHG[r], f4 = m, f5 = 1.f;
      w5 = w(5 + (int)e);
    }
    const float4 w0 = w(0), w1 = w(1), w2 = w(2), w3 = w(3), w4 = w(4);
    float v[4] = {x * w0.x + y * w1.x + z * w2.x + f3 * w3.x + f4 * w4.x + f5 * w5.x,
                  x * w0.y + y * w1.y + z * w2.y + f3 * w3.y + f4 * w4.y + f5 * w5.y,
                  x * w0.z + y * w1.z + z * w2.z + f3 * w3.z + f4 * w4.z + f5 * w5.z,
                  x * w0.w + y * w1.w + z * w2.w + f3 * w3.w + f4 * w4.w + f5 * w5.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) v[i] *= m;
    if (PROJ != nullptr) *reinterpret_cast<float4*>(PROJ + (size_t)r * C + 4 * lane) = make_float4(v[0], v[1], v[2], v[3]);
    if (ln) {
      float s = v[0] + v[1] + v[2] + v[3];
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
      const float mean = s * (1.f / C);
      float q = 0.f;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        v[i] -= mean;
        q += v[i] * v[i];
      }
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) q += __shfl_xor_sync(0xffffffffu, q, o);
      const float rs = 1.f / sqrtf(q * (1.f / C) + eps);
      const float4 g = *reinterpret_cast<const float4*>(LNW + 4 * lane), bb = *reinterpret_cast<const float4*>(LNB + 4 * lane);
      v[0] = v[0] * rs * g.x + bb.x, v[1] = v[1] * rs * g.y + bb.y, v[2] = v[2] * rs * g.z + bb.z, v[3] = v[3] * rs * g.w + bb.w;
    }
    if (out_bf16)
      *reinterpret_cast<uint2*>(reinterpret_cast<uint16_t*>(OUT) + (size_t)r * C + 4 * lane) = make_uint2(pack_bf16(v[0], v[1]), pack_bf16(v[2], v[3]));
    else
      *reinterpret_cast<float4*>(reinterpret_cast<float*>(OUT) + (size_t)r * C + 4 * lane) = make_float4(v[0], v[1], v[2], v[3]);
  }
}

// training backward of swa_embed_feat_sm100 (round 3 part 11), one warp per atom, lane = channels 4 lane .. 4 lane + 3:
//   ln: xhat = (p - mean) rstd from the saved masked projection p; dp = rstd (dy g - mean(dy g) - xhat mean(dy g xhat));
//       GB[0][r] = dy xhat, GB[1][r] = dy (the LayerNorm's weight / bias gradients are their column sums: torch.sum, fixed order)
//   no ln: dp = dy
//   DP[r] = m dp (fp32), F[r] = the row's features (fp32, nf = 6 legacy / 133 esmfold2: the weight gradient is DP^T F, one GEMM),
//   DPOS[r, k] = m sum_c DP[r, c] W[c, k] for k = 0 .. 2 (the positions are features 0 .. 2)
extern "C" __global__ void __launch_bounds__(256)
swa_embed_featbwd_sm100(const void* __restrict__ DY, const float* __restrict__ PROJ, const void* __restrict__ POS,
                        const long long* __restrict__ CHG, const long long* __restrict__ ELEM, const uint8_t* __restrict__ MASK,
                        const float* __restrict__ WT, const float* __restrict__ LNW, float* __restrict__ DP, float* __restrict__ GB,
                        float* __restrict__ F, void* __restrict__ DPOS, int M, int mode, int dy_bf16, int ln, float eps, int fl) {
  const int lane = threadIdx.x & 31;
  const int nf = mode == 0 ? 6 : 133;
  for (long long r = (long long)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5); r < M; r += (long long)gridDim.x * (blockDim.x >> 5)) {
    const float m = MASK[r] ? 1.f : 0.f;
    float d[4];
    if (dy_bf16) {
      const uint2 u = *reinterpret_cast<const uint2*>(reinterpret_cast<const uint16_t*>(DY) + (size_t)r * C + 4 * lane);
      d[0] = bf16lo(u.x), d[1] = bf16hi(u.x), d[2] = bf16lo(u.y), d[3] = bf16hi(u.y);
    } else {
      const float4 u = *reinterpret_cast<const float4*>(reinterpret_cast<const float*>(DY) + (size_t)r * C + 4 * lane);
      d[0] = u.x, d[1] = u.y, d[2] = u.z, d[3] = u.w;
    }
    if (ln) {
      const float4 p4 = *reinterpret_cast<const float4*>(PROJ + (size_t)r * C + 4 * lane);
      float v[4] = {p4.x, p4.y, p4.z, p4.w};
      float s = v[0] + v[1] + v[2] + v[3];
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
      const float mean = s * (1.f / C);
      float q = 0.f;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        v[i] -= mean;
        q += v[i] * v[i];
      }
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) q += __shfl_xor_sync(0xffffffffu, q, o);
      const float rs = 1.f / sqrtf(q * (1.f / C) + eps);
      const float4 g = *reinterpret_cast<const float4*>(LNW + 4 * lane);
      const float gg[4] = {g.x, g.y, g.z, g.w};
      float xh[4], dg[4], s1 = 0.f, s2 = 0.f;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        xh[i] = v[i] * rs;
        dg[i] = d[i] * gg[i];
        s1 += dg[i];
        s2 += dg[i] * xh[i];
      }
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) {
        s1 += __shfl_xor_sync(0xffffffffu, s1, o);
        s2 += __shfl_xor_sync(0xffffffffu, s2, o);
      }
      *reinterpret_cast<float4*>(GB + (size_t)r * C + 4 * lane) = make_float4(d[0] * xh[0], d[1] * xh[1], d[2] * xh[2], d[3] * xh[3]);
      *reinterpret_cast<float4*>(GB + ((size_t)M + r) * C + 4 * lane) = make_float4(d[0], d[1], d[2], d[3]);
#pragma unroll
      for (int i = 0; i < 4; ++i) d[i] = rs * (dg[i] - s1 * (1.f / C) - xh[i] * (s2 * (1.f / C)));
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) d[i] *= m;
    *reinterpret_cast<float4*>(DP + (size_t)r * C + 4 * lane) = make_float4(d[0], d[1], d[2], d[3]);
    float pg[3];
#pragma unroll
    for (int k = 0; k < 3; ++k) {
      const float4 w = ldw(WT, k, lane, nf, fl);
      pg[k] = d[0] * w.x + d[1] * w.y + d[2] * w.z + d[3] * w.w;
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) pg[k] += __shfl_xor_sync(0xffffffffu, pg[k], o);
    }
    if (lane < 3) {
      const float dv = m * pg[lane];
      if (fl & 4) {
        uint16_t h;
        asm("{ .reg .b16 t; cvt.rn.bf16.f32 t, %1; mov.b16 %0, t; }" : "=h"(h) : "f"(dv));
        reinterpret_cast<uint16_t*>(DPOS)[r * 3 + lane] = h;
      } else {
        reinterpret_cast<float*>(DPOS)[r * 3 + lane] = dv;
      }
    }
    // the features (as swa_embed_feat_sm100 makes them); none when F is null (the weight gradient by swa_embed_dwf_sm100)
    if (F == nullptr) continue;
    float* fr = F + (size_t)r * ((nf + 7) & ~7);                         // N2: rows padded to a multiple of 8 floats (16-B aligned GEMM operand)
    if (lane < 3) fr[lane] = ldpos(POS, r * 3 + lane, fl) * m;
    if (mode == 0) {
      if (lane == 3) fr[3] = m;
      else if (lane == 4) fr[4] = (float)ELEM[r] * m;
      else if (lane == 5) fr[5] = asinhf((float)CHG[r]) * m;
    } else {
      long long e = ELEM[r];
      e = e < 0 ? 0 : (e > 127 ? 127 : e);
      if (lane == 3) fr[3] = (float)CHG[r] * m;
      else if (lane == 4) fr[4] = m;
#pragma unroll
      for (int i = 0; i < 4; ++i) fr[5 + 4 * lane + i] = (4 * lane + i == (int)e) ? m : 0.f;
    }
  }
}

__device__ __forceinline__ void red_add(float* p, float v) { asm volatile("red.global.add.f32 [%0], %1;" :: "l"(p), "f"(v) : "memory"); }

// thread = (atom, 8 channels); T a multiple of 8. The thread of the FIRST atom of a run of equal token indices (within its row) sums the
// run's valid atoms in order and adds the sum once: with each token's atoms contiguous (the usual layout) every cell receives exactly one
// add -- the same bits run to run; a token split over several runs receives one add per run (correct, the order then the scheduler's).
extern "C" __global__ void __launch_bounds__(256)
swa_embed_scatter_sm100(const void* __restrict__ X, const uint8_t* __restrict__ MASK, const long long* __restrict__ IDX,
                        float* __restrict__ ACC, int B, int N, int ntok, int T, int relu, int x_bf16) {
  const int T8 = T / 8;
  const long long total = (long long)B * N * T8;
  for (long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x; t < total; t += (long long)gridDim.x * blockDim.x) {
    const long long m = t / T8;
    const int c8 = (int)(t - m * T8), b = (int)(m / N), s = (int)(m - (long long)b * N);
    const long long tok = IDX[m];
    if (s > 0 && IDX[m - 1] == tok) continue;                             // not the head of its run
    float sum[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f}, cnt = 0.f;
    for (long long j = m; j < (long long)(b + 1) * N && IDX[j] == tok; ++j) {
      if (!MASK[j]) continue;
      float v[8];
      if (x_bf16) {
        const uint4 u = *reinterpret_cast<const uint4*>(reinterpret_cast<const uint16_t*>(X) + (size_t)j * T + 8 * c8);
        const uint32_t w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
        for (int e = 0; e < 4; ++e) v[2 * e] = bf16lo(w[e]), v[2 * e + 1] = bf16hi(w[e]);
      } else {
        const float4* p = reinterpret_cast<const float4*>(reinterpret_cast<const float*>(X) + (size_t)j * T + 8 * c8);
        const float4 a = p[0], d = p[1];
        v[0] = a.x, v[1] = a.y, v[2] = a.z, v[3] = a.w, v[4] = d.x, v[5] = d.y, v[6] = d.z, v[7] = d.w;
      }
#pragma unroll
      for (int e = 0; e < 8; ++e) sum[e] += relu ? fmaxf(v[e], 0.f) : v[e];
      cnt += 1.f;
    }
    if (cnt == 0.f) continue;
    float* acc = ACC + ((size_t)b * ntok + (size_t)tok) * (T + 1);
#pragma unroll
    for (int e = 0; e < 8; ++e) red_add(acc + 8 * c8 + e, sum[e]);
    if (c8 == 0) red_add(acc + T, cnt);
  }
}

// N2: the read-out scatter with ONE thread per (group of 8 atoms, 8 channels) instead of per (atom, 8 channels): the head of a run is found
// inside the group (N a multiple of 8), the run's atoms are loaded in chunks of 8 (their token / mask loads together, then their data loads
// together: the run was one dependent load chain per atom), the sums in the same order as swa_embed_scatter_sm100 (the same bits). 1 / 8 of the
// threads, so one wave: ~4 us at 6144 atoms against 11.
extern "C" __global__ void __launch_bounds__(256)
swa_embed_scatter8_sm100(const void* __restrict__ X, const uint8_t* __restrict__ MASK, const long long* __restrict__ IDX,
                         float* __restrict__ ACC, int B, int N, int ntok, int T, int relu, int x_bf16) {
  const int T8 = T / 8;
  const long long total = (long long)B * (N / 8) * T8;
  for (long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x; t < total; t += (long long)gridDim.x * blockDim.x) {
    const long long g = t / T8;
    const int c8 = (int)(t - g * T8);
    const long long m0 = g * 8;
    const int b = (int)(m0 / N), s0 = (int)(m0 - (long long)b * N);
    const long long jend = (long long)(b + 1) * N;
    long long first = (s0 > 0) ? IDX[m0 - 1] : -1;
#pragma unroll 1
    for (int u = 0; u < 8; ++u) {
      const long long m = m0 + u, tok = IDX[m];
      const bool head = (u == 0 ? (s0 == 0 || first != tok) : IDX[m - 1] != tok);
      if (!head) continue;
      float sum[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f}, cnt = 0.f;
      bool go = true;
      for (long long j0 = m; go && j0 < jend; j0 += 8) {
        bool in[8];
        {
          bool prev = true;
#pragma unroll
          for (int w = 0; w < 8; ++w) {
            const bool ok = j0 + w < jend;
            const long long tj = ok ? IDX[j0 + w] : -1;
            const uint8_t mk = ok ? MASK[j0 + w] : 0;
            prev = prev && tj == tok;
            in[w] = prev && mk != 0;                                      // (the run is `prev`; its valid atoms `in`)
          }
          go = prev;
        }
        if (x_bf16) {
          uint4 xb[8];
#pragma unroll
          for (int w = 0; w < 8; ++w)
            if (in[w]) xb[w] = *reinterpret_cast<const uint4*>(reinterpret_cast<const uint16_t*>(X) + (size_t)(j0 + w) * T + 8 * c8);
#pragma unroll
          for (int w = 0; w < 8; ++w) {
            if (in[w]) {
              const uint32_t q[4] = {xb[w].x, xb[w].y, xb[w].z, xb[w].w};
#pragma unroll
              for (int e = 0; e < 4; ++e) {
                const float lo = bf16lo(q[e]), hi = bf16hi(q[e]);
                sum[2 * e] += relu ? fmaxf(lo, 0.f) : lo;
                sum[2 * e + 1] += relu ? fmaxf(hi, 0.f) : hi;
              }
              cnt += 1.f;
            }
          }
        } else {
          float4 xa[8], xd[8];
#pragma unroll
          for (int w = 0; w < 8; ++w)
            if (in[w]) {
              const float4* pp = reinterpret_cast<const float4*>(reinterpret_cast<const float*>(X) + (size_t)(j0 + w) * T + 8 * c8);
              xa[w] = pp[0];
              xd[w] = pp[1];
            }
#pragma unroll
          for (int w = 0; w < 8; ++w) {
            if (in[w]) {
              const float v[8] = {xa[w].x, xa[w].y, xa[w].z, xa[w].w, xd[w].x, xd[w].y, xd[w].z, xd[w].w};
#pragma unroll
              for (int e = 0; e < 8; ++e) sum[e] += relu ? fmaxf(v[e], 0.f) : v[e];
              cnt += 1.f;
            }
          }
        }
      }
      if (cnt == 0.f) continue;
      float* acc = ACC + ((size_t)b * ntok + (size_t)tok) * (T + 1);
#pragma unroll
      for (int e = 0; e < 8; ++e) red_add(acc + 8 * c8 + e, sum[e]);
      if (c8 == 0) red_add(acc + T, cnt);
    }
  }
}

extern "C" __global__ void __launch_bounds__(256)
swa_embed_tokmean_sm100(const float* __restrict__ ACC, void* __restrict__ OUT, long long rows, int T, int out_bf16) {
  const int T8 = T / 8;
  const long long total = rows * T8;
  for (long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x; t < total; t += (long long)gridDim.x * blockDim.x) {
    const long long r = t / T8;
    const int c8 = (int)(t - r * T8);
    const float* a = ACC + (size_t)r * (T + 1);
    const float cnt = fmaxf(a[T], 1.f);
    float v[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) v[e] = a[8 * c8 + e] / cnt;
    if (out_bf16)
      *reinterpret_cast<uint4*>(reinterpret_cast<uint16_t*>(OUT) + (size_t)r * T + 8 * c8) =
          make_uint4(pack_bf16(v[0], v[1]), pack_bf16(v[2], v[3]), pack_bf16(v[4], v[5]), pack_bf16(v[6], v[7]));
    else {
      float4* o = reinterpret_cast<float4*>(reinterpret_cast<float*>(OUT) + (size_t)r * T + 8 * c8);
      o[0] = make_float4(v[0], v[1], v[2], v[3]);
      o[1] = make_float4(v[4], v[5], v[6], v[7]);
    }
  }
}

// ---------------------------------------------------------------------------------------------- training read-out (round 3 part 11)
// one warp per atom, lane = channels 4 lane .. 4 lane + 3; x / y / dy bf16 or fp32 (flag), the arithmetic in fp32
__device__ __forceinline__ void ld4(const void* p, size_t off, int bf, float (&v)[4]) {
  if (bf) {
    const uint2 u = *reinterpret_cast<const uint2*>(reinterpret_cast<const uint16_t*>(p) + off);
    v[0] = bf16lo(u.x), v[1] = bf16hi(u.x), v[2] = bf16lo(u.y), v[3] = bf16hi(u.y);
  } else {
    const float4 u = *reinterpret_cast<const float4*>(reinterpret_cast<const float*>(p) + off);
    v[0] = u.x, v[1] = u.y, v[2] = u.z, v[3] = u.w;
  }
}
__device__ __forceinline__ void st4(void* p, size_t off, int bf, const float (&v)[4]) {
  if (bf) *reinterpret_cast<uint2*>(reinterpret_cast<uint16_t*>(p) + off) = make_uint2(pack_bf16(v[0], v[1]), pack_bf16(v[2], v[3]));
  else *reinterpret_cast<float4*>(reinterpret_cast<float*>(p) + off) = make_float4(v[0], v[1], v[2], v[3]);
}
// row statistics of the terminal norm: mode 1 LayerNorm (xhat = (x - mean) rs), 2 affine-free RMSNorm (xhat = x rs)
__device__ __forceinline__ float norm_hat(float (&v)[4], int mode, float eps) {
  if (mode == 1) {
    float s = v[0] + v[1] + v[2] + v[3];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    const float mean = s * (1.f / C);
#pragma unroll
    for (int i = 0; i < 4; ++i) v[i] -= mean;
  }
  float q = v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3];
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) q += __shfl_xor_sync(0xffffffffu, q, o);
  const float rs = 1.f / sqrtf(q * (1.f / C) + eps);
#pragma unroll
  for (int i = 0; i < 4; ++i) v[i] *= rs;
  return rs;
}
//   swa_embed_outnorm_sm100   y = norm(x) (LayerNorm: g, b) m, in x's dtype (the statements: the norm in fp32, rounded, times the mask)
extern "C" __global__ void __launch_bounds__(256)
swa_embed_outnorm_sm100(const void* __restrict__ X, const uint8_t* __restrict__ MASK, const float* __restrict__ LNW, const float* __restrict__ LNB,
                        void* __restrict__ Y, int M, int mode, int bf, float eps, float4* __restrict__ ZERO, int NZ) {
  for (int i = (int)(blockIdx.x * blockDim.x + threadIdx.x); i < NZ; i += (int)(gridDim.x * blockDim.x)) ZERO[i] = make_float4(0.f, 0.f, 0.f, 0.f);   // N2: the read-out's zeroed accumulator
  const int lane = threadIdx.x & 31;
  for (long long r = (long long)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5); r < M; r += (long long)gridDim.x * (blockDim.x >> 5)) {
    const float m = MASK[r] ? 1.f : 0.f;
    float v[4];
    ld4(X, (size_t)r * C + 4 * lane, bf, v);
    if (mode != 0) norm_hat(v, mode, eps);
    if (mode == 1) {
      const float4 g = *reinterpret_cast<const float4*>(LNW + 4 * lane), b = *reinterpret_cast<const float4*>(LNB + 4 * lane);
      v[0] = v[0] * g.x + b.x, v[1] = v[1] * g.y + b.y, v[2] = v[2] * g.z + b.z, v[3] = v[3] * g.w + b.w;
    }
    if (bf) {  // the statements round the norm's fp32 result to the activation dtype before the mask
#pragma unroll
      for (int i = 0; i < 4; ++i) v[i] = bf16lo(pack_bf16(v[i], 0.f));
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) v[i] *= m;
    st4(Y, (size_t)r * C + 4 * lane, bf, v);
  }
}
//   swa_embed_outnormbwd_sm100  dx of y = norm(x) m from dy (x saved); LayerNorm: GB[0][r] = dym xhat, GB[1][r] = dym (column sums: the
//                               weight / bias gradients); dx in dx_bf16 ? bf16 : fp32
extern "C" __global__ void __launch_bounds__(256)
swa_embed_outnormbwd_sm100(const void* __restrict__ DY, const void* __restrict__ X, const uint8_t* __restrict__ MASK, const float* __restrict__ LNW,
                           void* __restrict__ DX, float* __restrict__ GB, int M, int mode, int dy_bf, int x_bf, int dx_bf, float eps) {
  const int lane = threadIdx.x & 31;
  for (long long r = (long long)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5); r < M; r += (long long)gridDim.x * (blockDim.x >> 5)) {
    const float m = MASK[r] ? 1.f : 0.f;
    float v[4], d[4];
    ld4(X, (size_t)r * C + 4 * lane, x_bf, v);
    ld4(DY, (size_t)r * C + 4 * lane, dy_bf, d);
#pragma unroll
    for (int i = 0; i < 4; ++i) d[i] *= m;
    if (mode == 0) {                                                        // identity: dx = dy m
      st4(DX, (size_t)r * C + 4 * lane, dx_bf, d);
      continue;
    }
    const float rs = norm_hat(v, mode, eps);
    if (mode == 1) {
      *reinterpret_cast<float4*>(GB + (size_t)r * C + 4 * lane) = make_float4(d[0] * v[0], d[1] * v[1], d[2] * v[2], d[3] * v[3]);
      *reinterpret_cast<float4*>(GB + ((size_t)M + r) * C + 4 * lane) = make_float4(d[0], d[1], d[2], d[3]);
      const float4 g = *reinterpret_cast<const float4*>(LNW + 4 * lane);
      d[0] *= g.x, d[1] *= g.y, d[2] *= g.z, d[3] *= g.w;
    }
    float s1 = 0.f, s2 = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i) { s1 += d[i]; s2 += d[i] * v[i]; }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      s1 += __shfl_xor_sync(0xffffffffu, s1, o);
      s2 += __shfl_xor_sync(0xffffffffu, s2, o);
    }
    if (mode != 1) s1 = 0.f;                                               // RMSNorm: no mean term
#pragma unroll
    for (int i = 0; i < 4; ++i) d[i] = rs * (d[i] - s1 * (1.f / C) - v[i] * (s2 * (1.f / C)));
    st4(DX, (size_t)r * C + 4 * lane, dx_bf, d);
  }
}
//   swa_embed_tokgrad_sm100   the token mean's backward: dx[atom, :] = relu'(x) g[token, :] / max(count, 1) (thread = atom, 8 channels)
extern "C" __global__ void __launch_bounds__(256)
swa_embed_tokgrad_sm100(const void* __restrict__ G, const float* __restrict__ ACC, const long long* __restrict__ IDX, const void* __restrict__ X,
                        void* __restrict__ DX, int B, int N, int ntok, int T, int relu, int bf) {
  const int T8 = T / 8;
  const long long total = (long long)B * N * T8;
  for (long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x; t < total; t += (long long)gridDim.x * blockDim.x) {
    const long long m = t / T8;
    const int c8 = (int)(t - m * T8), b = (int)(m / N);
    const size_t tr = (size_t)b * ntok + (size_t)IDX[m];
    const float inv = 1.f / fmaxf(ACC[tr * (T + 1) + T], 1.f);
    float g[8], x[8];
    if (bf) {
      const uint4 u = *reinterpret_cast<const uint4*>(reinterpret_cast<const uint16_t*>(G) + tr * T + 8 * c8);
      const uint4 w = relu ? *reinterpret_cast<const uint4*>(reinterpret_cast<const uint16_t*>(X) + (size_t)m * T + 8 * c8) : make_uint4(0u, 0u, 0u, 0u);
      const uint32_t uu[4] = {u.x, u.y, u.z, u.w}, ww[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
      for (int e = 0; e < 4; ++e) g[2 * e] = bf16lo(uu[e]), g[2 * e + 1] = bf16hi(uu[e]), x[2 * e] = bf16lo(ww[e]), x[2 * e + 1] = bf16hi(ww[e]);
    } else {
      const float4* p = reinterpret_cast<const float4*>(reinterpret_cast<const float*>(G) + tr * T + 8 * c8);
      const float4* q = reinterpret_cast<const float4*>(reinterpret_cast<const float*>(X) + (size_t)m * T + 8 * c8);
      const float4 a = p[0], d = p[1], e0 = relu ? q[0] : a, e1 = relu ? q[1] : d;
      g[0] = a.x, g[1] = a.y, g[2] = a.z, g[3] = a.w, g[4] = d.x, g[5] = d.y, g[6] = d.z, g[7] = d.w;
      x[0] = e0.x, x[1] = e0.y, x[2] = e0.z, x[3] = e0.w, x[4] = e1.x, x[5] = e1.y, x[6] = e1.z, x[7] = e1.w;
    }
    float v[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) v[e] = (relu && !(x[e] > 0.f)) ? 0.f : g[e] * inv;
    if (bf)
      *reinterpret_cast<uint4*>(reinterpret_cast<uint16_t*>(DX) + (size_t)m * T + 8 * c8) =
          make_uint4(pack_bf16(v[0], v[1]), pack_bf16(v[2], v[3]), pack_bf16(v[4], v[5]), pack_bf16(v[6], v[7]));
    else {
      float4* o = reinterpret_cast<float4*>(reinterpret_cast<float*>(DX) + (size_t)m * T + 8 * c8);
      o[0] = make_float4(v[0], v[1], v[2], v[3]);
      o[1] = make_float4(v[4], v[5], v[6], v[7]);
    }
  }
}
// ---------------------------------------------------------------------------------------------- features' weight gradient (N2)
//   swa_embed_dwf_sm100   DW[c][j] = sum_r DP[r][c] F[r][j], F the features swa_embed_featbwd_sm100 used to write (nf = 6 legacy: all dense; 133
//                         esmfold2: 5 dense columns + the one-hot of the element, column 5 + e). One cluster of 8 CTAs (1024 threads) a column
//                         group; fixed order everywhere (the same bits run to run). Dense column j (group j): CTA k sums rows k ch .. k ch + ch - 1
//                         (thread (c, h): rows h, h + 8, ... of the chunk), the 8 h in order, then the 8 ranks in order over DSMEM. Hot group g
//                         (esmfold2): CTA k takes the element e = 8 g + k: the rows with that element (clamped to 0 .. 127, as the features), in row
//                         order, collected by a block-ordered compaction, thread (c, h) summing the matches h, h + 8, ... (8 loads in flight), the 8 h
//                         in order. M <= 8192 (the match list is static shared memory). Was a cuBLAS GEMM with K = M on 2 CTAs (11 us + a 3-us reduce).
extern "C" __global__ void __cluster_dims__(8, 1, 1) __launch_bounds__(1024)
swa_embed_dwf_sm100(const float* __restrict__ DP, const void* __restrict__ POS, const long long* __restrict__ CHG, const long long* __restrict__ ELEM,
                    const uint8_t* __restrict__ MASK, float* __restrict__ DW, int M, int mode, int fl) {
  __shared__ float part[8][128];
  __shared__ float tot[128];
  __shared__ float fv[1024];
  __shared__ int list[8192];
  __shared__ int wcnt[32];
  __shared__ int nlist;
  const int k = blockIdx.x & 7, grp = blockIdx.x >> 3;
  const int c = threadIdx.x & 127, h = threadIdx.x >> 7, lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int ndense = mode == 0 ? 6 : 5, nf = mode == 0 ? 6 : 133;
  if (grp < ndense) {
    const int j = grp, ch = (M + 7) / 8, i0 = min(M, k * ch), i1 = min(M, i0 + ch);
    constexpr int U = 16;                                                   // DP rows in flight a thread (loaded before the feature values: they overlap)
    const int nrow = (i1 - i0 + 7 - h) >> 3 > 0 ? (i1 - i0 - h + 7) >> 3 : 0;   // rows h, h + 8, .. of the chunk
    float v[U];
#pragma unroll
    for (int u = 0; u < U; ++u) v[u] = (u < nrow) ? DP[(size_t)(i0 + h + 8 * u) * C + c] : 0.f;
    for (int i = i0 + (int)threadIdx.x; i < i1; i += 1024) {
      const float m = MASK[i] ? 1.f : 0.f;
      float f;
      if (j < 3) f = ldpos(POS, (long long)i * 3 + j, fl) * m;
      else if (mode == 0) f = j == 3 ? m : (j == 4 ? (float)ELEM[i] * m : asinhf((float)CHG[i]) * m);
      else f = j == 3 ? (float)CHG[i] * m : m;
      fv[i - i0] = f;
    }
    __syncthreads();
    float a = 0.f;
    for (int t0 = 0; t0 < nrow; t0 += U) {
      if (t0 > 0) {
#pragma unroll
        for (int u = 0; u < U; ++u) v[u] = (t0 + u < nrow) ? DP[(size_t)(i0 + h + 8 * (t0 + u)) * C + c] : 0.f;
      }
#pragma unroll
      for (int u = 0; u < U; ++u)
        if (t0 + u < nrow) a = fmaf(v[u], fv[h + 8 * (t0 + u)], a);
    }
    part[h][c] = a;
    __syncthreads();
    if (h == 0) {
      float s = part[0][c];
#pragma unroll
      for (int q = 1; q < 8; ++q) s += part[q][c];
      tot[c] = s;
    }
    asm volatile("barrier.cluster.arrive.release.aligned; barrier.cluster.wait.acquire.aligned;" ::: "memory");
    if (k == 0 && h < 7) {                                                   // thread h fetches rank h + 1's total of channel c
      const uint32_t loc = (uint32_t)__cvta_generic_to_shared(&tot[c]);
      uint32_t ra;
      float v1;
      asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(ra) : "r"(loc), "r"(h + 1));
      asm volatile("ld.shared::cluster.f32 %0, [%1];" : "=f"(v1) : "r"(ra) : "memory");
      part[h + 1][c] = v1;
    }
    asm volatile("barrier.cluster.arrive.release.aligned; barrier.cluster.wait.acquire.aligned;" ::: "memory");   // peers stay until read
    if (k == 0 && h == 0) {
      float s = tot[c];
#pragma unroll
      for (int q = 1; q < 8; ++q) s += part[q][c];
      DW[(size_t)c * nf + j] = s;
    }
  } else {
    const int e = (grp - ndense) * 8 + k;
    if (threadIdx.x == 0) nlist = 0;
    __syncthreads();
    for (int base = 0; base < M; base += 1024) {
      const int r = base + (int)threadIdx.x;
      bool mt = false;
      if (r < M) {
        long long ev = ELEM[r];
        ev = ev < 0 ? 0 : (ev > 127 ? 127 : ev);
        mt = (int)ev == e;
      }
      const uint32_t b = __ballot_sync(0xffffffffu, mt);
      if (lane == 0) wcnt[warp] = __popc(b);
      __syncthreads();
      int off = nlist;
      for (int w = 0; w < warp; ++w) off += wcnt[w];
      if (mt) list[off + __popc(b & ((1u << lane) - 1u))] = r;
      __syncthreads();
      if (threadIdx.x == 0) {
        int t = 0;
        for (int w = 0; w < 32; ++w) t += wcnt[w];
        nlist += t;
      }
      __syncthreads();
    }
    const int n = nlist;
    float a = 0.f;
    int q = h;
    for (; q + 56 < n; q += 64) {
      float v[8];
#pragma unroll
      for (int u = 0; u < 8; ++u) v[u] = DP[(size_t)list[q + 8 * u] * C + c];
#pragma unroll
      for (int u = 0; u < 8; ++u) a += v[u];
    }
    for (; q < n; q += 8) a += DP[(size_t)list[q] * C + c];
    part[h][c] = a;
    __syncthreads();
    if (h == 0) {
      float s = part[0][c];
#pragma unroll
      for (int qq = 1; qq < 8; ++qq) s += part[qq][c];
      DW[(size_t)c * nf + 5 + e] = s;
    }
  }
}

// ---------------------------------------------------------------------------------------------- column sums (round 3 part 13)
//   swa_embed_colsum_sm100   OUT[r][c] = sum_i GB[r][i][c], i < M (the LayerNorm weight / bias gradients of featbwd / outnormbwd; was
//                            torch.sum over dim 1: a 11-us reduce_kernel + memset on the step's chain). Grid (R, 8), cluster (1, 8): CTA k sums
//                            rows k ch .. k ch + ch - 1 (1024 threads: 4 columns, rows g, g + 32, ... of the chunk), the 32 row groups in order in
//                            shared memory, rank 0 the 8 chunks in rank order over DSMEM -- a fixed order: the same bits run to run.
extern "C" __global__ void __cluster_dims__(1, 8, 1) __launch_bounds__(1024)
swa_embed_colsum_sm100(const float* __restrict__ GB, float* __restrict__ OUT, int M) {
  __shared__ float4 part[32][32];
  __shared__ float4 tot[32];
  const int r = blockIdx.x, k = blockIdx.y, c4 = threadIdx.x & 31, g = threadIdx.x >> 5;
  const int ch = (M + 7) / 8, i0 = min(M, k * ch), i1 = min(M, i0 + ch);
  const float4* src = reinterpret_cast<const float4*>(GB + (size_t)r * M * C) + c4;
  float4 a = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll 16
  for (int i = i0 + g; i < i1; i += 32) {
    const float4 v = __ldg(src + (size_t)i * (C / 4));
    a.x += v.x; a.y += v.y; a.z += v.z; a.w += v.w;
  }
  part[g][c4] = a;
  __syncthreads();
  if (g == 0) {
    float4 s = part[0][c4];
#pragma unroll
    for (int j = 1; j < 32; ++j) { const float4 v = part[j][c4]; s.x += v.x; s.y += v.y; s.z += v.z; s.w += v.w; }
    tot[c4] = s;
  }
  asm volatile("barrier.cluster.arrive.release.aligned; barrier.cluster.wait.acquire.aligned;" ::: "memory");
  if (k == 0 && g == 0) {
    float4 s = tot[c4];
    const uint32_t loc = (uint32_t)__cvta_generic_to_shared(&tot[c4]);
    for (int q = 1; q < 8; ++q) {
      uint32_t ra;
      float4 v;
      asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(ra) : "r"(loc), "r"(q));
      asm volatile("ld.shared::cluster.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "r"(ra) : "memory");
      s.x += v.x; s.y += v.y; s.z += v.z; s.w += v.w;
    }
    *reinterpret_cast<float4*>(OUT + (size_t)r * C + 4 * c4) = s;
  }
  asm volatile("barrier.cluster.arrive.release.aligned; barrier.cluster.wait.acquire.aligned;" ::: "memory");   // peers stay until read
}

extern "C" __global__ void __launch_bounds__(256)
swa_embed_rope_sm100(const void* __restrict__ POS, const long long* __restrict__ UID, const uint8_t* __restrict__ MASK,
                     const float* __restrict__ INV, const int* __restrict__ SRC, float* __restrict__ COS, float* __restrict__ SIN,
                     int* __restrict__ SEQ, int B, int N, int half, int nrope, int fl) {
  rope_body<false>(POS, UID, MASK, INV, SRC, COS, SIN, SEQ, B, N, half, nrope, fl);
}
extern "C" __global__ void __launch_bounds__(256)
swa_embed_rope2_sm100(const void* __restrict__ POS, const long long* __restrict__ UID, const uint8_t* __restrict__ MASK,
                      const float* __restrict__ INV, const int* __restrict__ SRC, float* __restrict__ COS, float* __restrict__ SIN,
                      int* __restrict__ SEQ, int B, int N, int half, int nrope, int fl) {
  rope_body<true>(POS, UID, MASK, INV, SRC, COS, SIN, SEQ, B, N, half, nrope, fl);
}

