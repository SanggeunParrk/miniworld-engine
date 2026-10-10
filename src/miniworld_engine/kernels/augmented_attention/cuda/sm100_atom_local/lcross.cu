// lcross.cu — the row kernels of the AF3 atom block's cross-attention mode (keys and values from a second AdaLN), sm_100a.
//
// AF3 / Protenix atom attention (cross_attention_mode): a = AdaLN_a(a, s) feeds q and the gate; the keys and values are projected from
//   xkv = AdaLN_kv(a, s) = LN(a) * sigmoid(Ws LN_g(s) + bs) + Wb LN_g(s)       (a the already normalised atoms, LN without affine)
// These kernels are the row-wise parts around cuBLAS GEMMs ([M, 128] rows, one warp per row, a lane holds 4 consecutive channels):
//   local_cond_ln        cn = LN(c) * gamma                                   (the conditioning's LayerNorm, bf16 out)
//   local_kv_fwd         xkv from x1, the modulation GEMM output mkv [M, 256] = [pre-sigmoid scale | shift] and the scale bias
//   local_kv_bwd         dx1, dmkv from dxkv
//   local_adaln1_extra   the extra gradient dx1 through AdaLN 1 (+= into ds, dmod's scale / shift blocks, the scale-bias column sums)
//   local_cond_ln_bwd    dc and dgamma of local_cond_ln
// SPDX-License-Identifier: Apache-2.0
#include "local.cuh"

DEVI float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o >= 1; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}
DEVI void ld4(const __nv_bfloat16* p, float* x) {
  const uint2 u = *reinterpret_cast<const uint2*>(p);
  x[0] = __uint_as_float(u.x << 16); x[1] = __uint_as_float(u.x & 0xffff0000u);
  x[2] = __uint_as_float(u.y << 16); x[3] = __uint_as_float(u.y & 0xffff0000u);
}
DEVI void st4(__nv_bfloat16* p, const float* x) {
  *reinterpret_cast<uint2*>(p) = make_uint2(pack_bf16(x[0], x[1]), pack_bf16(x[2], x[3]));
}
// x -> (x - mean) * rstd over the warp's 128 channels
DEVI float norm128(float* x, float eps) {
  const float mean = warp_sum(x[0] + x[1] + x[2] + x[3]) * (1.f / 128.f);
  float v = 0.f;
#pragma unroll
  for (int i = 0; i < 4; ++i) { x[i] -= mean; v += x[i] * x[i]; }
  const float rstd = 1.f / sqrtf(warp_sum(v) * (1.f / 128.f) + eps);
#pragma unroll
  for (int i = 0; i < 4; ++i) x[i] *= rstd;
  return rstd;
}
DEVI float sigm(float x) { return 1.f / (1.f + __expf(-x)); }

extern "C" __global__ void __launch_bounds__(256)
local_cond_ln(const __nv_bfloat16* __restrict__ c, const float* __restrict__ gamma, __nv_bfloat16* __restrict__ cn, int M, float eps) {
  const int row = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  if (row >= M) return;
  float x[4];
  ld4(c + (size_t)row * DM + lane * 4, x);
  norm128(x, eps);
#pragma unroll
  for (int i = 0; i < 4; ++i) x[i] *= gamma[lane * 4 + i];
  st4(cn + (size_t)row * DM + lane * 4, x);
}

extern "C" __global__ void __launch_bounds__(256)
local_kv_fwd(const __nv_bfloat16* __restrict__ x1, const __nv_bfloat16* __restrict__ mkv, const float* __restrict__ bs,
             __nv_bfloat16* __restrict__ xkv, int M, float eps) {
  const int row = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  if (row >= M) return;
  float x[4], pre[4], sh[4];
  ld4(x1 + (size_t)row * DM + lane * 4, x);
  ld4(mkv + (size_t)row * 2 * DM + lane * 4, pre);
  ld4(mkv + (size_t)row * 2 * DM + DM + lane * 4, sh);
  norm128(x, eps);
#pragma unroll
  for (int i = 0; i < 4; ++i) x[i] = x[i] * sigm(pre[i] + bs[lane * 4 + i]) + sh[i];
  st4(xkv + (size_t)row * DM + lane * 4, x);
}

extern "C" __global__ void __launch_bounds__(256)
local_kv_bwd(const __nv_bfloat16* __restrict__ x1, const __nv_bfloat16* __restrict__ mkv, const float* __restrict__ bs,
             const __nv_bfloat16* __restrict__ dxkv, __nv_bfloat16* __restrict__ dx1, __nv_bfloat16* __restrict__ dmkv, int M, float eps) {
  const int row = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  if (row >= M) return;
  float x[4], pre[4], dy[4];
  ld4(x1 + (size_t)row * DM + lane * 4, x);
  ld4(mkv + (size_t)row * 2 * DM + lane * 4, pre);
  ld4(dxkv + (size_t)row * DM + lane * 4, dy);
  const float rstd = norm128(x, eps);                                  // x = LN(x1)
  float sk[4], dpre[4], dxn[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    sk[i] = sigm(pre[i] + bs[lane * 4 + i]);
    dpre[i] = dy[i] * x[i] * sk[i] * (1.f - sk[i]);
    dxn[i] = dy[i] * sk[i];
  }
  const float m1 = warp_sum(dxn[0] + dxn[1] + dxn[2] + dxn[3]) * (1.f / 128.f);
  const float m2 = warp_sum(dxn[0] * x[0] + dxn[1] * x[1] + dxn[2] * x[2] + dxn[3] * x[3]) * (1.f / 128.f);
  float o[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) o[i] = rstd * (dxn[i] - m1 - x[i] * m2);
  st4(dx1 + (size_t)row * DM + lane * 4, o);
  st4(dmkv + (size_t)row * 2 * DM + lane * 4, dpre);
  st4(dmkv + (size_t)row * 2 * DM + DM + lane * 4, dy);
}

// ds += AdaLN1 backward of dx1 (the gradient the key / value branch sends to x1); dmod[:, 0:128] += d pre-sigmoid scale, dmod[:, 128:256] += dx1
// Four rows per warp in flight: every load of the four rows is issued before the first LayerNorm reduction (a row's chain -- load, two
// shuffle reductions, three more loads, stores -- took ~2.5 us and each warp walked ~40 rows of them in sequence: 181 us for ~30 us of traffic).

// the block's eight warps' per-lane partial sums -> one atomic per channel per block (a warp each made 4736 atomics onto the same 128
// addresses, serialised in L2: ~150 us of the ~180 these two kernels took)
DEVI void block_add4(float* __restrict__ out, const float (&acc)[4]) {
  __shared__ float4 red[8][32];
  const int w = threadIdx.x >> 5, lane = threadIdx.x & 31;
  red[w][lane] = make_float4(acc[0], acc[1], acc[2], acc[3]);
  __syncthreads();
  if (w == 0) {
    float4 s = red[0][lane];
#pragma unroll
    for (int i = 1; i < 8; ++i) { const float4 v = red[i][lane]; s.x += v.x; s.y += v.y; s.z += v.z; s.w += v.w; }
    atomicAdd(out + lane * 4, s.x); atomicAdd(out + lane * 4 + 1, s.y); atomicAdd(out + lane * 4 + 2, s.z); atomicAdd(out + lane * 4 + 3, s.w);
  }
}
constexpr int UNR = 4;
extern "C" __global__ void __launch_bounds__(256)
local_adaln1_extra(const __nv_bfloat16* __restrict__ a, const __nv_bfloat16* __restrict__ mod, const __nv_bfloat16* __restrict__ dx1,
                   __nv_bfloat16* __restrict__ ds, __nv_bfloat16* __restrict__ dmod, float* __restrict__ dbias, int M, float eps) {
  const int lane = threadIdx.x & 31;
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
  const int step = gridDim.x * 8;
  for (int row0 = blockIdx.x * 8 + (threadIdx.x >> 5); row0 < M; row0 += UNR * step) {
    float x[UNR][4], s1[UNR][4], d[UNR][4], t[UNR][4], u[UNR][4], v[UNR][4];
#pragma unroll
    for (int j = 0; j < UNR; ++j) {
      const int row = row0 + j * step;
      if (row < M) {
        ld4(a + (size_t)row * DM + lane * 4, x[j]);
        ld4(mod + (size_t)row * 6 * DM + lane * 4, s1[j]);                // rn(sigmoid) scale of AdaLN 1
        ld4(dx1 + (size_t)row * DM + lane * 4, d[j]);
        ld4(ds + (size_t)row * DM + lane * 4, t[j]);
        ld4(dmod + (size_t)row * 6 * DM + lane * 4, u[j]);
        ld4(dmod + (size_t)row * 6 * DM + DM + lane * 4, v[j]);
      }
    }
#pragma unroll
    for (int j = 0; j < UNR; ++j) {
      const int row = row0 + j * step;
      if (row >= M) break;
      const float rstd = norm128(x[j], eps);
      float dl[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        dl[i] = d[j][i] * s1[j][i];
        const float dpre = d[j][i] * x[j][i] * s1[j][i] * (1.f - s1[j][i]);
        acc[i] += dpre;
        u[j][i] += dpre;
      }
      const float m1 = warp_sum(dl[0] + dl[1] + dl[2] + dl[3]) * (1.f / 128.f);
      const float m2 = warp_sum(dl[0] * x[j][0] + dl[1] * x[j][1] + dl[2] * x[j][2] + dl[3] * x[j][3]) * (1.f / 128.f);
#pragma unroll
      for (int i = 0; i < 4; ++i) { t[j][i] += rstd * (dl[i] - m1 - x[j][i] * m2); v[j][i] += d[j][i]; }
      st4(ds + (size_t)row * DM + lane * 4, t[j]);
      st4(dmod + (size_t)row * 6 * DM + lane * 4, u[j]);
      st4(dmod + (size_t)row * 6 * DM + DM + lane * 4, v[j]);
    }
  }
  block_add4(dbias, acc);
}

// dc = LN backward of (dcn * gamma); dgamma += sum over rows of dcn * LN(c)
extern "C" __global__ void __launch_bounds__(256)
local_cond_ln_bwd(const __nv_bfloat16* __restrict__ c, const __nv_bfloat16* __restrict__ dcn, const float* __restrict__ gamma,
                  __nv_bfloat16* __restrict__ dc, float* __restrict__ dgamma, int M, float eps) {
  const int lane = threadIdx.x & 31;
  float acc[4] = {0.f, 0.f, 0.f, 0.f}, g[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) g[i] = gamma[lane * 4 + i];
  const int step = gridDim.x * 8;
  for (int row0 = blockIdx.x * 8 + (threadIdx.x >> 5); row0 < M; row0 += UNR * step) {
    float x[UNR][4], d[UNR][4];
#pragma unroll
    for (int j = 0; j < UNR; ++j) {
      const int row = row0 + j * step;
      if (row < M) {
        ld4(c + (size_t)row * DM + lane * 4, x[j]);
        ld4(dcn + (size_t)row * DM + lane * 4, d[j]);
      }
    }
#pragma unroll
    for (int j = 0; j < UNR; ++j) {
      const int row = row0 + j * step;
      if (row >= M) break;
      const float rstd = norm128(x[j], eps);
      float dl[4], o[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) { acc[i] += d[j][i] * x[j][i]; dl[i] = d[j][i] * g[i]; }
      const float m1 = warp_sum(dl[0] + dl[1] + dl[2] + dl[3]) * (1.f / 128.f);
      const float m2 = warp_sum(dl[0] * x[j][0] + dl[1] * x[j][1] + dl[2] * x[j][2] + dl[3] * x[j][3]) * (1.f / 128.f);
#pragma unroll
      for (int i = 0; i < 4; ++i) o[i] = rstd * (dl[i] - m1 - x[j][i] * m2);
      st4(dc + (size_t)row * DM + lane * 4, o);
    }
  }
  block_add4(dgamma, acc);
}

// column sums of a bf16 [M, 128] view (row pitch ld) in fp32: G partial rows (grid-stride over rows, four loads in flight per warp), summed by the host
extern "C" __global__ void __launch_bounds__(256)
local_colsum(const __nv_bfloat16* __restrict__ x, int ld, float* __restrict__ part, int M) {
  const int w = threadIdx.x >> 5, lane = threadIdx.x & 31;
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
  const int step = gridDim.x * 8;
  for (int row0 = blockIdx.x * 8 + w; row0 < M; row0 += UNR * step) {
    float v[UNR][4];
#pragma unroll
    for (int j = 0; j < UNR; ++j) {
      const int row = row0 + j * step;
      if (row < M) ld4(x + (size_t)row * ld + lane * 4, v[j]);
      else { v[j][0] = v[j][1] = v[j][2] = v[j][3] = 0.f; }
    }
#pragma unroll
    for (int j = 0; j < UNR; ++j)
#pragma unroll
      for (int i = 0; i < 4; ++i) acc[i] += v[j][i];
  }
  __shared__ float4 red[8][32];
  red[w][lane] = make_float4(acc[0], acc[1], acc[2], acc[3]);
  __syncthreads();
  if (w == 0) {
    float4 s = red[0][lane];
#pragma unroll
    for (int i = 1; i < 8; ++i) { const float4 q = red[i][lane]; s.x += q.x; s.y += q.y; s.z += q.z; s.w += q.w; }
    *reinterpret_cast<float4*>(part + (size_t)blockIdx.x * 128 + lane * 4) = s;
  }
}

