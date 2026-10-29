// lcross.cu — the row kernels of the AF3 atom block's cross-attention mode (keys and values from a second AdaLN), sm_100a.
//
// AF3 / Protenix atom attention (cross_attention_mode): a = AdaLN_a(a, s) feeds q and the gate; the keys and values are projected from
//   xkv = AdaLN_kv(a, s) = LN(a) * sigmoid(Ws LN_g(s) + bs) + Wb LN_g(s)       (a the already normalised atoms, LN without affine)
// The K / V AdaLN itself and its backward run inside pre_fwd.cu / pre_bwd.cu (-DCROSS=1). These kernels are the row-wise parts around cuBLAS
// GEMMs ([M, 128] bf16 rows), those of the conditioning's LayerNorm:
//   local_cond_ln        cn = LN(c) * gamma                                   (the conditioning's LayerNorm, bf16 out)
//   local_cond_ln_bwd    dc (optionally added onto a given dc) and dgamma of local_cond_ln
// sm_90a (ATOM_HOPPER) additionally: local_cond_ln_bwd_sm90 (dgamma as per-block partials), local_kv_fwd_sm90 / local_kv_bwd_sm90 /
// local_adaln1_extra_sm90 (the K / V AdaLN, which on Hopper is not inside pre_fwd / pre_bwd) and the small parameter-gradient kernels.
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

#if defined(ATOM_HOPPER)
// Hopper: the same reduction, but the block's sum is stored (fixed order, no atomics) into out[128] of its own row of a [grid, 128] table
DEVI void block_partial8(float* __restrict__ out, const float (&acc)[8]) {
  __shared__ float red[16][128];
  const int hl = threadIdx.x & 15, half = threadIdx.x >> 4;
#pragma unroll
  for (int i = 0; i < 8; ++i) red[half][hl * 8 + i] = acc[i];
  __syncthreads();
  if (threadIdx.x < 128) {
    float s = 0.f;
#pragma unroll
    for (int k = 0; k < 16; ++k) s += red[k][threadIdx.x];
    out[threadIdx.x] = s;
  }
}
#endif

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
template <int U, bool PARTIAL = false>
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
#if defined(ATOM_HOPPER)
  if constexpr (PARTIAL) block_partial8(dgamma + (size_t)blockIdx.x * DM, acc);   // dgamma: the [grid, 128] partial table
  else
#endif
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
// Hopper: dgamma without atomics. A block stores its sum into row blockIdx.x of the [gridDim.x, 128] fp32 table ``part``, and
// local_reduce_columns_sm90 adds the rows in a fixed order (deterministic dgamma).
extern "C" __global__ void __launch_bounds__(256)
local_cond_ln_bwd_sm90(const __nv_bfloat16* __restrict__ c, const __nv_bfloat16* __restrict__ dcn, const float* __restrict__ gamma,
                       __nv_bfloat16* dc, const __nv_bfloat16* dcin, float* __restrict__ part, int M, float eps) {
  cond_ln_bwd_body<1, true>(c, dcn, gamma, dc, dcin, part, M, eps);
}

// The K / V AdaLN of the cross-attention mode as separate row kernels (one warp per row, a lane holds 4 consecutive channels): on Hopper
// pre_fwd / pre_bwd carry no K / V AdaLN.
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
DEVI float wnorm128(float* x, float eps) {                            // x -> (x - mean) * rstd over the warp's 128 channels
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

// xkv = LN(x1) * sigmoid(mkv[:, :128] + bs) + mkv[:, 128:]
extern "C" __global__ void __launch_bounds__(256)
local_kv_fwd_sm90(const __nv_bfloat16* __restrict__ x1, const __nv_bfloat16* __restrict__ mkv, const float* __restrict__ bs,
                  __nv_bfloat16* __restrict__ xkv, int M, float eps) {
  const int row = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  if (row >= M) return;
  float x[4], pre[4], sh[4];
  ld4(x1 + (size_t)row * DM + lane * 4, x);
  ld4(mkv + (size_t)row * 2 * DM + lane * 4, pre);
  ld4(mkv + (size_t)row * 2 * DM + DM + lane * 4, sh);
  wnorm128(x, eps);
#pragma unroll
  for (int i = 0; i < 4; ++i) x[i] = x[i] * sigm(pre[i] + bs[lane * 4 + i]) + sh[i];
  st4(xkv + (size_t)row * DM + lane * 4, x);
}

// dx1 and dmkv = [d pre-sigmoid scale | d shift] from dxkv
extern "C" __global__ void __launch_bounds__(256)
local_kv_bwd_sm90(const __nv_bfloat16* __restrict__ x1, const __nv_bfloat16* __restrict__ mkv, const float* __restrict__ bs,
                  const __nv_bfloat16* __restrict__ dxkv, __nv_bfloat16* __restrict__ dx1, __nv_bfloat16* __restrict__ dmkv, int M,
                  float eps) {
  const int row = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  if (row >= M) return;
  float x[4], pre[4], dy[4];
  ld4(x1 + (size_t)row * DM + lane * 4, x);
  ld4(mkv + (size_t)row * 2 * DM + lane * 4, pre);
  ld4(dxkv + (size_t)row * DM + lane * 4, dy);
  const float rstd = wnorm128(x, eps);                                 // x = LN(x1)
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

// ds += AdaLN 1 backward of dx1 (the gradient the K / V branch sends to x1); dmod[:, 0:128] += d pre-sigmoid scale, dmod[:, 128:256] += dx1.
// The scale-bias column sum (dbias) is not accumulated here: finish_columns_hopper reduces it from dmod.
extern "C" __global__ void __launch_bounds__(256)
local_adaln1_extra_sm90(const __nv_bfloat16* __restrict__ a, const __nv_bfloat16* __restrict__ mod, const __nv_bfloat16* __restrict__ dx1,
                        __nv_bfloat16* __restrict__ ds, __nv_bfloat16* __restrict__ dmod, float* __restrict__ dbias, int M, float eps) {
  const int lane = threadIdx.x & 31;
  for (int row = blockIdx.x * 8 + (threadIdx.x >> 5); row < M; row += gridDim.x * 8) {
    float x[4], s1[4], d[4];
    ld4(a + (size_t)row * DM + lane * 4, x);
    ld4(mod + (size_t)row * 6 * DM + lane * 4, s1);                   // rn(sigmoid) scale of AdaLN 1
    ld4(dx1 + (size_t)row * DM + lane * 4, d);
    const float rstd = wnorm128(x, eps);
    float dl[4], dpre[4], t[4], u[4], v[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) { dl[i] = d[i] * s1[i]; dpre[i] = d[i] * x[i] * s1[i] * (1.f - s1[i]); }
    const float m1 = warp_sum(dl[0] + dl[1] + dl[2] + dl[3]) * (1.f / 128.f);
    const float m2 = warp_sum(dl[0] * x[0] + dl[1] * x[1] + dl[2] * x[2] + dl[3] * x[3]) * (1.f / 128.f);
    ld4(ds + (size_t)row * DM + lane * 4, t);
    ld4(dmod + (size_t)row * 6 * DM + lane * 4, u);
    ld4(dmod + (size_t)row * 6 * DM + DM + lane * 4, v);
#pragma unroll
    for (int i = 0; i < 4; ++i) { t[i] += rstd * (dl[i] - m1 - x[i] * m2); u[i] += dpre[i]; v[i] += d[i]; }
    st4(ds + (size_t)row * DM + lane * 4, t);
    st4(dmod + (size_t)row * 6 * DM + lane * 4, u);
    st4(dmod + (size_t)row * 6 * DM + DM + lane * 4, v);
  }
}

// Fixed-order partial reductions preserve the cross-KV parameter gradients.
// column sums of src[:, 0:128] (row stride 256): block b sums rows b, b + G, ... (16 rows in flight, a thread holds 8 columns); the 16 row
// groups of a block combine in a fixed order through shared memory -> part[b, 128]
extern "C" __global__ void __launch_bounds__(256) local_kv_bias_part_sm90(const __nv_bfloat16* src, float* part, int M) {
 __shared__ float red[16][129];
 const int c8 = (threadIdx.x & 15) * 8, rg = threadIdx.x >> 4;
 float v[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
 for (int r = blockIdx.x * 16 + rg; r < M; r += gridDim.x * 16) {
  const uint4 u = *reinterpret_cast<const uint4*>(src + size_t(r) * 256 + c8);
  const uint32_t w[4] = {u.x, u.y, u.z, u.w};
  #pragma unroll
  for (int i = 0; i < 4; ++i) { v[2 * i] += __uint_as_float(w[i] << 16); v[2 * i + 1] += __uint_as_float(w[i] & 0xffff0000u); }
 }
 #pragma unroll
 for (int i = 0; i < 8; ++i) red[rg][c8 + i] = v[i];
 __syncthreads();
 if (threadIdx.x < 128) {
  float s = 0.f;
  #pragma unroll
  for (int g = 0; g < 16; ++g) s += red[g][threadIdx.x];
  part[size_t(blockIdx.x) * 128 + threadIdx.x] = s;
 }
}
extern "C" __global__ void local_reduce_columns_sm90(const float* part,float* out,int count) {
 int c=threadIdx.x;
 float value=0.f;
 for(int r=0;r<count;++r)value+=part[size_t(r)*128+c];
 out[c]=value;
}
extern "C" __global__ void local_zero_kv_sm90(__nv_bfloat16* dp,int M) {
 for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<size_t(M)*256;i+=size_t(gridDim.x)*blockDim.x)
  dp[(i/256)*512+128+i%256]=__float2bfloat16_rn(0.f);
}
extern "C" __global__ void local_add_cond_sm90(__nv_bfloat16* dc,const __nv_bfloat16* extra,int M) {
 for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<size_t(M)*128;i+=size_t(gridDim.x)*blockDim.x)
  dc[i]=__float2bfloat16_rn(__bfloat162float(dc[i])+__bfloat162float(extra[i]));
}
extern "C" __global__ void local_cross_grad_pack_sm90(const float* g0,const float* g1,const float* g2,const float* g3,
 void* o0,void* o1,void* o2,void* o3,int bfmask) {
 for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<33024;i+=gridDim.x*blockDim.x){
  const int j=i<128?0:i<16512?1:i<16640?2:3;
  const int offset=i-(j==0?0:j==1?128:j==2?16512:16640);
  const float* src=j==0?g0:j==1?g1:j==2?g2:g3;
  void* dst=j==0?o0:j==1?o1:j==2?o2:o3;
  if(bfmask&(1<<j))reinterpret_cast<__nv_bfloat16*>(dst)[offset]=__float2bfloat16_rn(src[offset]);
  else reinterpret_cast<float*>(dst)[offset]=src[offset];
 }
}
extern "C" __global__ void local_kv_pack_sm90(const void* gamma,const void* ws,const void* bias,const void* wb,
 __nv_bfloat16* wm,float* bs,float* gg,int bfmask) {
 int i=blockIdx.x*blockDim.x+threadIdx.x;
 auto val=[&](const void* p,int j,int n){return (bfmask&(1<<n))?__bfloat162float(reinterpret_cast<const __nv_bfloat16*>(p)[j]):reinterpret_cast<const float*>(p)[j];};
 if(i<32768)wm[i]=__float2bfloat16_rn(val(i<16384?ws:wb,i%16384,i<16384?1:3));
 if(i<128){bs[i]=val(bias,i,2);gg[i]=val(gamma,i,0);}
}
#endif

// local_sample_sum: out[i, c] = sum over the samples a = 0 .. A - 1, in that order, of x[a n + i, c] (bf16 [A n, cols], rows ldx apart; fp32
// sums, bf16 out, rows ldo apart): the gradient of a conditioning table the samples share (the row kernels write it per sample row). A thread
// per (row, 8 columns); deterministic.
extern "C" __global__ void __launch_bounds__(256)
local_sample_sum(const __nv_bfloat16* __restrict__ x, int ldx, __nv_bfloat16* __restrict__ out, int ldo, int A, int n, int cols) {
  const int groups = cols / 8;
  const long long total = (long long)n * groups;
  for (long long e = (long long)blockIdx.x * blockDim.x + threadIdx.x; e < total; e += (long long)gridDim.x * blockDim.x) {
    const int i = (int)(e / groups), g = (int)(e - (long long)i * groups);
    float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    const __nv_bfloat16* p = x + (size_t)i * ldx + 8 * g;
#pragma unroll 4
    for (int a = 0; a < A; ++a) {
      const uint4 u = *reinterpret_cast<const uint4*>(p + (size_t)a * n * ldx);
      const uint32_t w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
      for (int k = 0; k < 4; ++k) { acc[2 * k] += __uint_as_float(w[k] << 16); acc[2 * k + 1] += __uint_as_float(w[k] & 0xffff0000u); }
    }
    *reinterpret_cast<uint4*>(out + (size_t)i * ldo + 8 * g) =
        make_uint4(pack_bf16(acc[0], acc[1]), pack_bf16(acc[2], acc[3]), pack_bf16(acc[4], acc[5]), pack_bf16(acc[6], acc[7]));
  }
}

// local_table_sum: the shared conditioning's table gradient out [mrows, ldo] (bf16) from the row kernels' partials (sm100_atom/tsum.cuh):
// out[16 t + r, c] = the sum over the slots of table tile t of part[t][j][r][c'] (fp32, in slot order: deterministic). Up to three sources
// (p, C, split, col0, col1; grid y; p null: none): column c' of a source goes to col0 + c' (c' < split) or col1 + c' - split. A thread
// per (table row, 8 columns). The slots of tile t are the warpgroup slots u = owner(t A) .. owner(t A + A - 1) whose range is not
// empty (lo(u) = floor(ntile u / U), owner(g) = floor(((g + 1) U - 1) / ntile)), slot j = u - owner(t A).
DEVI int ts_lo(int v, int U, int ntile) { return (int)(((long long)ntile * v) / U); }
DEVI int ts_owner(int g, int U, int ntile) { return (int)((((long long)g + 1) * U - 1) / ntile); }
extern "C" __global__ void __launch_bounds__(256)
local_table_sum(const float* __restrict__ p0, int C0, int sp0, int a0, int b0, const float* __restrict__ p1, int C1, int sp1, int a1, int b1,
                const float* __restrict__ p2, int C2, int sp2, int a2, int b2, __nv_bfloat16* __restrict__ out, int ldo, int mrows, int A, int U,
                int ntile, int J) {
  const int src = blockIdx.y;                                           // the source: grid y
  const float* __restrict__ p = src == 0 ? p0 : src == 1 ? p1 : p2;
  const int C = src == 0 ? C0 : src == 1 ? C1 : C2, sp = src == 0 ? sp0 : src == 1 ? sp1 : sp2;
  const int ca = src == 0 ? a0 : src == 1 ? a1 : a2, cb = src == 0 ? b0 : src == 1 ? b1 : b2;
  if (p == nullptr) return;
  const int groups = C >> 3, total = mrows * groups;                    // (< 2^31)
  for (int e = blockIdx.x * blockDim.x + threadIdx.x; e < total; e += gridDim.x * blockDim.x) {
    const int row = e / groups, c = 8 * (e - row * groups), t = row >> 4, r = row & 15;
    const int u0 = ts_owner(t * A, U, ntile), u1 = ts_owner(t * A + A - 1, U, ntile);
    float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    for (int u = u0; u <= u1; ++u) {
      if (ts_lo(u + 1, U, ntile) == ts_lo(u, U, ntile)) continue;     // (an empty range: no tile, no partial)
      const float4* s = reinterpret_cast<const float4*>(p + ((((size_t)t * J + (u - u0)) * 16 + r) * C + c));
      const float4 x = s[0], y = s[1];
      acc[0] += x.x; acc[1] += x.y; acc[2] += x.z; acc[3] += x.w; acc[4] += y.x; acc[5] += y.y; acc[6] += y.z; acc[7] += y.w;
    }
    *reinterpret_cast<uint4*>(out + (size_t)row * ldo + (c < sp ? ca + c : cb + c - sp)) =
        make_uint4(pack_bf16(acc[0], acc[1]), pack_bf16(acc[2], acc[3]), pack_bf16(acc[4], acc[5]), pack_bf16(acc[6], acc[7]));
  }
}
