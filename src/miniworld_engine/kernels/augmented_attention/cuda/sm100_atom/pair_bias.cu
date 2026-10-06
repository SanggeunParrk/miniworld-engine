// pair_bias.cu — the atom DiT's pair-bias producer and its backward, memory-bound CUDA (one element (i, j) per thread-iteration).
//
//   forward:  bias[h, i, j] = rstd (z[i, j, :] . W'[h] - mean sum_c W'[h, c]),  W' = gamma * Wb   (LayerNorm without offset, eps 1e-5)
//             written head-major [H, N, N] and transposed [H, N(key), N(query)] (staged through shared memory, so both stores coalesce)
//   backward: dz = rstd (dxh - mean(dxh) - xh mean(dxh xh)),  dxh = sum_h dbias[h] W'[h];   d(W')[h, c] = sum_ij dbias[h] xh[c]
//             (per-block partials, summed on the host side)
// Key mask and padding: the bias side is N x N (N a multiple of 128, the attention's length), z is NZ x NZ (NZ <= N atoms);
// kv [NZ] (bytes 0 / 1: the caller's bool mask as is; null = every key valid) marks the valid keys; keys >= NZ are padding
// (pair_bias_fwd_pad, the bounds-checked build). A masked key's
// column is written as MASKED (-1e4, bf16 -9984: 2^(-9984 log2 e + score - max) is 0 in fp32 for any row with a valid key, and
// small enough that scores and the LSE keep their bits, so the backward's recomputed P matches the forward's even in a row
// whose keys are all masked -- there the weights are the softmax of the scores, where the module's finfo.min fill gives
// uniform ones); padded query rows (i >= NZ) are 0 on valid keys. The backward reads only i, j < NZ and drops dbias on masked
// keys (the module's masked_fill passes no gradient there).
// Block tile: 32 rows (i) x 64 columns (j), 256 threads: thread (ty, tx) = (tid / 64, tid % 64) takes rows ty + 4 k, k < 8.
// SPDX-License-Identifier: Apache-2.0
#include <cstdint>
#include <cuda_bf16.h>

#if defined(ATOM_HOPPER)
// The module reference uses a finite masked_fill: no valid keys means
// uniform attention, with zero Q/K/bias gradients and mean(dO) for every V.
// The main attention uses -inf for masked columns; this deterministic edge
// epilogue supplies mean(V) or mean(dO) only when the live mask is all false.
extern "C" __global__ void atom_empty_uniform_sm90(
    const unsigned char* keys, const __nv_bfloat16* src, __nv_bfloat16* dst,
    int N, int dst_stride, int dst_offset) {
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  __shared__ int valid_warp[8];
  int valid = 0;
  for (int j = tid; j < N; j += 256) valid |= int(keys[j] != 0);
  for (int d = 16; d; d >>= 1) valid |= __shfl_down_sync(0xffffffffu, valid, d);
  if (lane == 0) valid_warp[warp] = valid;
  __syncthreads();
  valid = 0;
  for (int w = 0; w < 8; ++w) valid |= valid_warp[w];
  if (valid) return;

  __shared__ float partial[8][32];
  const int col = blockIdx.x * 32 + lane, sample = blockIdx.y;
  float sum = 0.f;
  for (int j = warp; j < N; j += 8)
    sum += __bfloat162float(src[(size_t(sample) * N + j) * 128 + col]);
  partial[warp][lane] = sum;
  __syncthreads();
  sum = 0.f;
  for (int w = 0; w < 8; ++w) sum += partial[w][lane];
  const __nv_bfloat16 mean = __float2bfloat16_rn(sum / float(N));
  for (int j = warp; j < N; j += 8)
    dst[(size_t(sample) * N + j) * dst_stride + dst_offset + col] = mean;
}
#endif

#if defined(ATOM_HOPPER)
extern "C" __global__ void atom_pair_partials_sm90(const float* src,float* dst,int nb) {
 __shared__ float s[16][16];
 int x=threadIdx.x&15,y=threadIdx.x>>4,c=blockIdx.x*16+x,shard=blockIdx.y;
 int lo=int((long long)nb*shard/16),hi=int((long long)nb*(shard+1)/16);float sum=0;
 for(int j=lo+y;j<hi;j+=16)sum+=src[size_t(j)*64+c];
 s[y][x]=sum;__syncthreads();
 if(y==0){float v=0;for(int j=0;j<16;++j)v+=s[j][x];dst[shard*64+c]=v;}
}
#endif

#if defined(ATOM_HOPPER)
extern "C" __global__ void atom_pair_weight_sm90(const void* wb,const void* gamma,float* out,int bfmask) {
 int c=threadIdx.x;
 if(c<64){float w=(bfmask&1)?__bfloat162float(reinterpret_cast<const __nv_bfloat16*>(wb)[c]):reinterpret_cast<const float*>(wb)[c];
 float g=(bfmask&2)?__bfloat162float(reinterpret_cast<const __nv_bfloat16*>(gamma)[c&15]):reinterpret_cast<const float*>(gamma)[c&15];
 out[c]=w*g;}
}
extern "C" __global__ void atom_pair_finish_sm90(const float* part,const void* wb,const void* gamma,
 float* dg,float* dw,int nb,int bfmask) {
 __shared__ float x[64];
 int c=threadIdx.x;float sum=0;
 for(int j=0;j<nb;++j)sum+=part[size_t(j)*64+c];
 float g=(bfmask&2)?__bfloat162float(reinterpret_cast<const __nv_bfloat16*>(gamma)[c&15]):reinterpret_cast<const float*>(gamma)[c&15];
 float w=(bfmask&1)?__bfloat162float(reinterpret_cast<const __nv_bfloat16*>(wb)[c]):reinterpret_cast<const float*>(wb)[c];
 dw[c]=sum*g;x[c]=sum*w;__syncthreads();
 if(c<16)dg[c]=((x[c]+x[16+c])+x[32+c])+x[48+c];
}
#endif

#if defined(ATOM_HOPPER)
extern "C" __global__ void atom_dbias_transpose_sum_sm90(const float* src,float* dst,int N,int A){
 __shared__ float tile[32][33];
 int h=blockIdx.z,q0=blockIdx.x*32,k0=blockIdx.y*32,x=threadIdx.x,y=threadIdx.y;
 #pragma unroll
 for(int j=0;j<32;j+=8){
   float v=0;for(int a=0;a<A;++a)v+=src[((size_t(a)*4+h)*N+k0+y+j)*N+q0+x];
   tile[y+j][x]=v;
 }
 __syncthreads();
 #pragma unroll
 for(int j=0;j<32;j+=8)dst[(size_t(h)*N+q0+y+j)*N+k0+x]=tile[x][y+j];
}
extern "C" __global__ void atom_pack_dqkv_sm90(const __nv_bfloat16* q,const __nv_bfloat16* k,
 const __nv_bfloat16* v,__nv_bfloat16* dp,int M){
 int i=blockIdx.x*blockDim.x+threadIdx.x;
 for(;i<M*64;i+=blockDim.x*gridDim.x){
   int row=i/64,c=2*(i%64);
   const uint32_t a=*reinterpret_cast<const uint32_t*>(q+size_t(row)*128+c);
   const uint32_t b=*reinterpret_cast<const uint32_t*>(k+size_t(row)*128+c);
   const uint32_t d=*reinterpret_cast<const uint32_t*>(v+size_t(row)*128+c);
   *reinterpret_cast<uint32_t*>(dp+size_t(row)*512+c)=a;
   *reinterpret_cast<uint32_t*>(dp+size_t(row)*512+128+c)=b;
   *reinterpret_cast<uint32_t*>(dp+size_t(row)*512+256+c)=d;
 }
}
#endif

constexpr int C = 16, H = 4, TI = 32, TJ = 64, NT = 256;
#if defined(ATOM_HOPPER_TRUE_MASK)
// Opt-in Hopper correction for the efficient-SDPA all-masked reference.
// The natural attention core records a nonfinite LSE for an empty row and
// its backward already returns zero probability for those rows.
constexpr float MASKED = -__builtin_inff();
#else
constexpr float MASKED = -1e4f;
#endif

__device__ __forceinline__ void load_row(const __nv_bfloat16* z, size_t e, float (&x)[C]) {
  const uint4* p = reinterpret_cast<const uint4*>(z + e * C);
  const uint4 a = __ldg(p), b = __ldg(p + 1);
  const uint32_t w[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
#pragma unroll
  for (int k = 0; k < 8; ++k) { x[2 * k] = __uint_as_float(w[k] << 16); x[2 * k + 1] = __uint_as_float(w[k] & 0xffff0000u); }
}

// PAD: N > NZ (padded atoms: bounds checks on the z reads); without padding the loop is branch-free, the mask a select.
template <bool PAD>
__device__ __forceinline__ void pair_bias_fwd_body(const __nv_bfloat16* __restrict__ z, const float* __restrict__ wp,
                                                   __nv_bfloat16* __restrict__ bo, __nv_bfloat16* __restrict__ bt, int N, int NZ,
                                                   const unsigned char* __restrict__ kv, float eps) {
  __shared__ float w[H * C], ws[H];
  __shared__ __nv_bfloat16 st[H][TJ][TI + 2];
  const int tid = threadIdx.x, tx = tid & 63, ty = tid >> 6;
  if (tid < H * C) w[tid] = wp[tid];
  __syncthreads();
  if (tid < H) { float s = 0.f; for (int c = 0; c < C; ++c) s += w[tid * C + c]; ws[tid] = s; }
  __syncthreads();
  const int i0 = blockIdx.y * TI, j = blockIdx.x * TJ + tx;
  const bool jin = !PAD || j < NZ;
  const bool valid = jin && (kv == nullptr || kv[j]);
#pragma unroll 2
  for (int k = 0; k < TI / 4; ++k) {
    const int il = ty + 4 * k, i = i0 + il;
    float y[H];
    if (!PAD || (i < NZ && jin)) {
      float x[C];
      load_row(z, (size_t)i * NZ + j, x);
      float s = 0.f, s2 = 0.f;
#pragma unroll
      for (int c = 0; c < C; ++c) { s += x[c]; s2 += x[c] * x[c]; }
      const float mean = s * (1.f / C), rstd = rsqrtf(fmaxf(s2 * (1.f / C) - mean * mean, 0.f) + eps);
#pragma unroll
      for (int h = 0; h < H; ++h) {
        float d = 0.f;
#pragma unroll
        for (int c = 0; c < C; ++c) d = fmaf(x[c], w[h * C + c], d);
        y[h] = rstd * (d - mean * ws[h]);
      }
    } else {                                                               // a padded query row or key
#pragma unroll
      for (int h = 0; h < H; ++h) y[h] = 0.f;
    }
#pragma unroll
    for (int h = 0; h < H; ++h) {
      const __nv_bfloat16 v = __float2bfloat16_rn(valid ? y[h] : MASKED);
      bo[((size_t)h * N + i) * N + j] = v;
      st[h][tx][il] = v;
    }
  }
  if (bt == nullptr) return;
  __syncthreads();
  // transposed: rows j0 .. j0 + 63 of bias^T[h], 32 queries (64 B) each; a thread writes 4 bf16 pairs per head
  const int j0 = blockIdx.x * TJ;
#pragma unroll
  for (int h = 0; h < H; ++h)
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const int idx = q * NT + tid, r = idx >> 4, pr = idx & 15;                                     // row r (0..63), pair pr (0..15)
      const __nv_bfloat162 v = __halves2bfloat162(st[h][r][2 * pr], st[h][r][2 * pr + 1]);
      *reinterpret_cast<__nv_bfloat162*>(bt + ((size_t)h * N + j0 + r) * N + i0 + 2 * pr) = v;
    }
}

extern "C" __global__ void __launch_bounds__(NT)
pair_bias_fwd(const __nv_bfloat16* __restrict__ z, const float* __restrict__ wp, __nv_bfloat16* __restrict__ bo,
              __nv_bfloat16* __restrict__ bt, int N, int NZ, const unsigned char* __restrict__ kv, float eps) {
  pair_bias_fwd_body<false>(z, wp, bo, bt, N, NZ, kv, eps);
}

extern "C" __global__ void __launch_bounds__(NT)
pair_bias_fwd_pad(const __nv_bfloat16* __restrict__ z, const float* __restrict__ wp, __nv_bfloat16* __restrict__ bo,
                  __nv_bfloat16* __restrict__ bt, int N, int NZ, const unsigned char* __restrict__ kv, float eps) {
  pair_bias_fwd_body<true>(z, wp, bo, bt, N, NZ, kv, eps);
}

extern "C" __global__ void __launch_bounds__(NT)
pair_bias_bwd(const __nv_bfloat16* __restrict__ z, const float* __restrict__ wp, const float* __restrict__ db,
              __nv_bfloat16* __restrict__ dz, float* __restrict__ pw, int N, int NZ, const unsigned char* __restrict__ kv, float eps) {
  __shared__ float w[H * C];
  __shared__ float red[NT / 32][H * C];
  const int tid = threadIdx.x, tx = tid & 63, ty = tid >> 6;
  if (tid < H * C) w[tid] = wp[tid];
  __syncthreads();
  const int i0 = blockIdx.y * TI, j = blockIdx.x * TJ + tx;
  float acc[H][C];
#pragma unroll
  for (int h = 0; h < H; ++h)
#pragma unroll
    for (int c = 0; c < C; ++c) acc[h][c] = 0.f;
  const bool valid = j < NZ && (kv == nullptr || kv[j]);
  for (int k = 0; k < TI / 4; ++k) {
    const int i = i0 + ty + 4 * k;
    if (i >= NZ || j >= NZ) continue;                                      // padding: no z there
    const size_t e = (size_t)i * NZ + j;
    float x[C];
    load_row(z, e, x);
    float s = 0.f, s2 = 0.f;
#pragma unroll
    for (int c = 0; c < C; ++c) { s += x[c]; s2 += x[c] * x[c]; }
    const float mean = s * (1.f / C), rstd = rsqrtf(fmaxf(s2 * (1.f / C) - mean * mean, 0.f) + eps);
    float g[H];
#pragma unroll
    for (int h = 0; h < H; ++h) g[h] = valid ? __ldg(db + ((size_t)h * N + i) * N + j) : 0.f;
    float m1 = 0.f, m2 = 0.f, dxh[C];
#pragma unroll
    for (int c = 0; c < C; ++c) {
      const float xh = (x[c] - mean) * rstd;
      float d = 0.f;
#pragma unroll
      for (int h = 0; h < H; ++h) { d = fmaf(g[h], w[h * C + c], d); acc[h][c] = fmaf(g[h], xh, acc[h][c]); }
      dxh[c] = d; m1 += d; m2 += d * xh; x[c] = xh;
    }
    m1 *= 1.f / C; m2 *= 1.f / C;
    uint32_t o[8];
#pragma unroll
    for (int k2 = 0; k2 < 8; ++k2) {
      const __nv_bfloat162 v = __floats2bfloat162_rn((dxh[2 * k2] - m1 - x[2 * k2] * m2) * rstd, (dxh[2 * k2 + 1] - m1 - x[2 * k2 + 1] * m2) * rstd);
      o[k2] = *reinterpret_cast<const uint32_t*>(&v);
    }
    uint4* p = reinterpret_cast<uint4*>(dz + e * C);
    p[0] = make_uint4(o[0], o[1], o[2], o[3]);
    p[1] = make_uint4(o[4], o[5], o[6], o[7]);
  }
  // block partial of d(W'): warp shuffle, then across the 8 warps
#pragma unroll
  for (int h = 0; h < H; ++h)
#pragma unroll
    for (int c = 0; c < C; ++c) {
      float v = acc[h][c];
#pragma unroll
      for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
      if ((tid & 31) == 0) red[tid >> 5][h * C + c] = v;
    }
  __syncthreads();
  if (tid < H * C) {
    float v = 0.f;
#pragma unroll
    for (int r = 0; r < NT / 32; ++r) v += red[r][tid];
    pw[((size_t)blockIdx.y * gridDim.x + blockIdx.x) * H * C + tid] = v;
  }
}
