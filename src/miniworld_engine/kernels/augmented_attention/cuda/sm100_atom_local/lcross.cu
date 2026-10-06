// lcross.cu — the row kernels of the AF3 atom block's cross-attention mode (keys and values from a second AdaLN), sm_100a.
//
// AF3 / Protenix atom attention (cross_attention_mode): a = AdaLN_a(a, s) feeds q and the gate; the keys and values are projected from
//   xkv = AdaLN_kv(a, s) = LN(a) * sigmoid(Ws LN_g(s) + bs) + Wb LN_g(s)       (a the already normalised atoms, LN without affine)
// The K / V AdaLN itself and its backward run inside pre_fwd.cu / pre_bwd.cu (-DCROSS=1). These kernels are the row-wise parts around cuBLAS
// GEMMs ([M, 128] bf16 rows), those of the conditioning's LayerNorm:
//   local_cond_ln        cn = LN(c) * gamma                                   (the conditioning's LayerNorm, bf16 out)
//   local_cond_ln_bwd    dc (optionally added onto a given dc) and dgamma of local_cond_ln
//
// Layout: a persistent grid (8 blocks of 256 threads per SM), a row per half-warp, 16 B (8 channels) per lane: a row is one 256-B
// request of 16 lanes, the LayerNorm reductions are 4 shuffle steps, and the loads of U rows are issued before the first reduction.
// (One row per warp with 8-B lanes and one block per 8 rows ran at 40-55 % of the copy rate: cond_ln 33 -> 19 us at 196608 rows.)
// The bias / gamma gradients are reduced in the block and end in one atomic per channel per block. U = 1: more rows in flight per half-warp
// cost registers, and with them blocks per SM, and ran slower (cond_ln_bwd 35 -> 43-51 us).
// SPDX-License-Identifier: Apache-2.0
#include "local.cuh"

DEVI float hsum(float v) {                                            // sum over the 16 lanes of a row
#pragma unroll
  for (int o = 8; o >= 1; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}
DEVI void ld8(const __nv_bfloat16* p, float* x) {
  const uint4 u = *reinterpret_cast<const uint4*>(p);
  const uint32_t w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
  for (int e = 0; e < 4; ++e) { x[2 * e] = __uint_as_float(w[e] << 16); x[2 * e + 1] = __uint_as_float(w[e] & 0xffff0000u); }
}
// out-of-range rows load zeros: every lane of the warp keeps taking part in the half-warp shuffles, and zero rows add nothing to the sums
DEVI void ld8z(const __nv_bfloat16* p, float* x, bool ok) {
  if (ok) ld8(p, x);
  else {
#pragma unroll
    for (int i = 0; i < 8; ++i) x[i] = 0.f;
  }
}
DEVI void st8(__nv_bfloat16* p, const float* x) {
  *reinterpret_cast<uint4*>(p) = make_uint4(pack_bf16(x[0], x[1]), pack_bf16(x[2], x[3]), pack_bf16(x[4], x[5]), pack_bf16(x[6], x[7]));
}
// x -> (x - mean) * rstd over the row's 128 channels (16 lanes x 8)
DEVI float norm128(float* x, float eps) {
  float s = 0.f;
#pragma unroll
  for (int i = 0; i < 8; ++i) s += x[i];
  const float mean = hsum(s) * (1.f / 128.f);
  float v = 0.f;
#pragma unroll
  for (int i = 0; i < 8; ++i) { x[i] -= mean; v += x[i] * x[i]; }
  const float rstd = rsqrtf(hsum(v) * (1.f / 128.f) + eps);
#pragma unroll
  for (int i = 0; i < 8; ++i) x[i] *= rstd;
  return rstd;
}

// the block's 16 half-warps' per-lane partial sums [8 channels] -> out[128] += (one atomic per channel per block)
DEVI void block_reduce8(float* __restrict__ out, const float (&acc)[8]) {
  __shared__ float red[16][128];
  const int hl = threadIdx.x & 15, half = threadIdx.x >> 4;
#pragma unroll
  for (int i = 0; i < 8; ++i) red[half][hl * 8 + i] = acc[i];
  __syncthreads();
  if (threadIdx.x < 128) {
    float s = 0.f;
#pragma unroll
    for (int k = 0; k < 16; ++k) s += red[k][threadIdx.x];
    atomicAdd(out + threadIdx.x, s);
  }
}

template <int U>
DEVI void cond_ln_body(const __nv_bfloat16* __restrict__ c, const float* __restrict__ gamma, __nv_bfloat16* __restrict__ cn, int M, float eps) {
  const int hl = threadIdx.x & 15, step = gridDim.x * 16;
  float g[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) g[i] = gamma[hl * 8 + i];
  for (int row0 = blockIdx.x * 16 + (threadIdx.x >> 4); row0 < M; row0 += U * step) {
    float x[U][8];
#pragma unroll
    for (int j = 0; j < U; ++j) ld8z(c + (size_t)(row0 + j * step) * DM + hl * 8, x[j], row0 + j * step < M);
#pragma unroll
    for (int j = 0; j < U; ++j) {
      const int row = row0 + j * step;
      norm128(x[j], eps);
#pragma unroll
      for (int i = 0; i < 8; ++i) x[j][i] *= g[i];
      if (row < M) st8(cn + (size_t)row * DM + hl * 8, x[j]);
    }
  }
}

// dc = LN backward of (dcn * gamma) (+ dcin in bf16 when given: the gradient of c's other consumer; dc may be dcin); dgamma += sum rows dcn * LN(c)
template <int U>
DEVI void cond_ln_bwd_body(const __nv_bfloat16* __restrict__ c, const __nv_bfloat16* __restrict__ dcn, const float* __restrict__ gamma,
                           __nv_bfloat16* dc, const __nv_bfloat16* dcin, float* __restrict__ dgamma, int M, float eps) {
  const int hl = threadIdx.x & 15, step = gridDim.x * 16;
  float acc[8], g[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) { acc[i] = 0.f; g[i] = gamma[hl * 8 + i]; }
  for (int row0 = blockIdx.x * 16 + (threadIdx.x >> 4); row0 < M; row0 += U * step) {
    float x[U][8], d[U][8], pr[U][8];
#pragma unroll
    for (int j = 0; j < U; ++j) {
      const int row = row0 + j * step;
      ld8z(c + (size_t)row * DM + hl * 8, x[j], row < M);
      ld8z(dcn + (size_t)row * DM + hl * 8, d[j], row < M);
      if (dcin) ld8z(dcin + (size_t)row * DM + hl * 8, pr[j], row < M);
    }
#pragma unroll
    for (int j = 0; j < U; ++j) {
      const int row = row0 + j * step;
      const float rstd = norm128(x[j], eps);
      float dl[8], o[8], s1 = 0.f, s2 = 0.f;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        acc[i] += d[j][i] * x[j][i];
        dl[i] = d[j][i] * g[i];
        s1 += dl[i];
        s2 += dl[i] * x[j][i];
      }
      const float m1 = hsum(s1) * (1.f / 128.f), m2 = hsum(s2) * (1.f / 128.f);
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        o[i] = rstd * (dl[i] - m1 - x[j][i] * m2);
        if (dcin) o[i] = __bfloat162float(__float2bfloat16(o[i])) + pr[j][i];
      }
      if (row < M) st8(dc + (size_t)row * DM + hl * 8, o);
    }
  }
  block_reduce8(dgamma, acc);
}

extern "C" __global__ void __launch_bounds__(256)
local_cond_ln(const __nv_bfloat16* __restrict__ c, const float* __restrict__ gamma, __nv_bfloat16* __restrict__ cn, int M, float eps) {
  cond_ln_body<1>(c, gamma, cn, M, eps);
}
extern "C" __global__ void __launch_bounds__(256)
local_cond_ln_bwd(const __nv_bfloat16* __restrict__ c, const __nv_bfloat16* __restrict__ dcn, const float* __restrict__ gamma,
                  __nv_bfloat16* dc, const __nv_bfloat16* dcin, float* __restrict__ dgamma, int M, float eps) {
  cond_ln_bwd_body<1>(c, dcn, gamma, dc, dcin, dgamma, M, eps);
}

#if defined(ATOM_HOPPER)
extern "C" __global__ void local_kv_pack_sm90(const void* gamma,const void* ws,const void* bias,const void* wb,
 __nv_bfloat16* wm,float* bs,float* gg,int bfmask) {
 int i=blockIdx.x*blockDim.x+threadIdx.x;
 auto val=[&](const void* p,int j,int n){return (bfmask&(1<<n))?__bfloat162float(reinterpret_cast<const __nv_bfloat16*>(p)[j]):reinterpret_cast<const float*>(p)[j];};
 if(i<32768)wm[i]=__float2bfloat16_rn(val(i<16384?ws:wb,i%16384,i<16384?1:3));
 if(i<128){bs[i]=val(bias,i,2);gg[i]=val(gamma,i,0);}
}
#endif
