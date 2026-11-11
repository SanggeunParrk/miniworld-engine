// aux_sm80.cuh -- the small kernels around the attention core of the A100 (sm_80) attention-with-shared-bias path: the pair bias's producer and its backward
// (the atom width: 16 pair channels, 4 heads) and the backward's row term delta.  All memory-bound CUDA, no tensor cores.
//
// pair_bias_fwd:  bias[h, i, j] = oscale rstd (z[i, j, :] . W'[h] - mean sum_c W'[h, c])        W' = gamma * Wb   (LayerNorm without offset, eps 1e-5)
//                 written head-major [H, N, N] (bf16 for bf16 z with oscale = sqrt(head dim): the attention core takes its bias in raw units, bias / sm_scale;
//                 fp32 for fp32 z with oscale = 1: the TF32 core takes natural units); the key mask (kv: one
//                 byte per key, null = every key valid) and the padding (N > NZ: the attention's length is the next multiple of 128) are folded in: a masked or
//                 padded key's column is MASKED (-1e4 natural units: a score that 2^(.) turns into 0 for any row with a valid key and that keeps the log-sum-exp
//                 finite), a padded query row is 0.
// pair_bias_bwd:  dz = rstd (dxh - mean(dxh) - xh mean(dxh xh)),  dxh = sum_h dbias[h] W'[h];   d(W')[h, c] = sum_ij dbias[h] xh[c]   (per-block partials,
//                 summed by the caller in a fixed order).  It reads only i, j < NZ and drops dbias on masked keys (the module's masked_fill passes no gradient).
// attn_delta:     delta[a][h][i] = sum_d dO[a, i, h, d] O[a, i, h, d]   (the attention backward's row term; O bf16 or fp32).
//
// The pair bias kernels follow the B200 atom DiT's (``sm100_atom/pair_bias.cu``, plain CUDA): one element (i, j) per thread-iteration, a block tile of 32 rows x
// 64 columns, 256 threads.
#pragma once
#include <cuda_bf16.h>
#include <stdint.h>

namespace aa80 {

constexpr int PB_C = 16, PB_H = 4, PB_TI = 32, PB_TJ = 64, PB_NT = 256;
constexpr float PB_MASKED = -1e4f;

__device__ __forceinline__ void pb_load_row(const __nv_bfloat16* z, size_t e, float (&x)[PB_C]) {
  const uint4* p = reinterpret_cast<const uint4*>(z + e * PB_C);
  const uint4 a = __ldg(p), b = __ldg(p + 1);
  const uint32_t w[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
#pragma unroll
  for (int k = 0; k < 8; ++k) { x[2 * k] = __uint_as_float(w[k] << 16); x[2 * k + 1] = __uint_as_float(w[k] & 0xffff0000u); }
}

__device__ __forceinline__ void pb_load_row(const float* z, size_t e, float (&x)[PB_C]) {
  const float4* p = reinterpret_cast<const float4*>(z + e * PB_C);
#pragma unroll
  for (int k = 0; k < 4; ++k) { const float4 a = __ldg(p + k); x[4 * k] = a.x; x[4 * k + 1] = a.y; x[4 * k + 2] = a.z; x[4 * k + 3] = a.w; }
}
__device__ __forceinline__ void pb_store(__nv_bfloat16* p, float v) { *p = __float2bfloat16_rn(v); }
__device__ __forceinline__ void pb_store(float* p, float v) { *p = v; }
__device__ __forceinline__ void pb_store_row(__nv_bfloat16* p, const float (&v)[PB_C]) {            // 16 values as two 16-byte stores
  uint32_t o[8];
#pragma unroll
  for (int k2 = 0; k2 < 8; ++k2) { const __nv_bfloat162 t = __floats2bfloat162_rn(v[2 * k2], v[2 * k2 + 1]); o[k2] = *reinterpret_cast<const uint32_t*>(&t); }
  uint4* q = reinterpret_cast<uint4*>(p);
  q[0] = make_uint4(o[0], o[1], o[2], o[3]);
  q[1] = make_uint4(o[4], o[5], o[6], o[7]);
}
__device__ __forceinline__ void pb_store_row(float* p, const float (&v)[PB_C]) {
  float4* q = reinterpret_cast<float4*>(p);
#pragma unroll
  for (int k = 0; k < 4; ++k) q[k] = make_float4(v[4 * k], v[4 * k + 1], v[4 * k + 2], v[4 * k + 3]);
}

// PAD: N > NZ (padded atoms: bounds checks on the z reads); without padding the loop is branch-free, the mask a select.
template <bool PAD, typename TI, typename TO>
__global__ void __launch_bounds__(PB_NT) pair_bias_fwd_kernel(const TI* __restrict__ z, const float* __restrict__ wp, TO* __restrict__ bo,
                                                              int N, int NZ, const unsigned char* __restrict__ kv, float eps, float oscale, float fill) {
  __shared__ float w[PB_H * PB_C], ws[PB_H];
  const int tid = threadIdx.x, tx = tid & 63, ty = tid >> 6;
  if (tid < PB_H * PB_C) w[tid] = wp[tid];
  __syncthreads();
  if (tid < PB_H) { float s = 0.f; for (int c = 0; c < PB_C; ++c) s += w[tid * PB_C + c]; ws[tid] = s; }
  __syncthreads();
  const int i0 = blockIdx.y * PB_TI, j = blockIdx.x * PB_TJ + tx;
  const bool jin = !PAD || j < NZ;
  const bool valid = jin && (kv == nullptr || kv[j]);
#pragma unroll 2
  for (int k = 0; k < PB_TI / 4; ++k) {
    const int i = i0 + ty + 4 * k;
    float y[PB_H];
    if (!PAD || (i < NZ && jin)) {
      float x[PB_C];
      pb_load_row(z, (size_t)i * NZ + j, x);
      float s = 0.f, s2 = 0.f;
#pragma unroll
      for (int c = 0; c < PB_C; ++c) { s += x[c]; s2 += x[c] * x[c]; }
      const float mean = s * (1.f / PB_C), rstd = rsqrtf(fmaxf(s2 * (1.f / PB_C) - mean * mean, 0.f) + eps);
#pragma unroll
      for (int h = 0; h < PB_H; ++h) {
        float d = 0.f;
#pragma unroll
        for (int c = 0; c < PB_C; ++c) d = fmaf(x[c], w[h * PB_C + c], d);
        y[h] = rstd * (d - mean * ws[h]);
      }
    } else {                                                               // a padded query row or key
#pragma unroll
      for (int h = 0; h < PB_H; ++h) y[h] = 0.f;
    }
#pragma unroll
    for (int h = 0; h < PB_H; ++h) pb_store(bo + ((size_t)h * N + i) * N + j, valid ? y[h] * oscale : fill);
  }
}

template <typename T>
__global__ void __launch_bounds__(PB_NT) pair_bias_bwd_kernel(const T* __restrict__ z, const float* __restrict__ wp, const float* __restrict__ db,
                                                              T* __restrict__ dz, float* __restrict__ pw, int N, int NZ,
                                                              const unsigned char* __restrict__ kv, float eps) {
  __shared__ float w[PB_H * PB_C];
  __shared__ float red[PB_NT / 32][PB_H * PB_C];
  const int tid = threadIdx.x, tx = tid & 63, ty = tid >> 6;
  if (tid < PB_H * PB_C) w[tid] = wp[tid];
  __syncthreads();
  const int i0 = blockIdx.y * PB_TI, j = blockIdx.x * PB_TJ + tx;
  float acc[PB_H][PB_C];
#pragma unroll
  for (int h = 0; h < PB_H; ++h)
#pragma unroll
    for (int c = 0; c < PB_C; ++c) acc[h][c] = 0.f;
  const bool valid = j < NZ && (kv == nullptr || kv[j]);
  for (int k = 0; k < PB_TI / 4; ++k) {
    const int i = i0 + ty + 4 * k;
    if (i >= NZ || j >= NZ) continue;                                      // padding: no z there
    const size_t e = (size_t)i * NZ + j;
    float x[PB_C];
    pb_load_row(z, e, x);
    float s = 0.f, s2 = 0.f;
#pragma unroll
    for (int c = 0; c < PB_C; ++c) { s += x[c]; s2 += x[c] * x[c]; }
    const float mean = s * (1.f / PB_C), rstd = rsqrtf(fmaxf(s2 * (1.f / PB_C) - mean * mean, 0.f) + eps);
    float g[PB_H];
#pragma unroll
    for (int h = 0; h < PB_H; ++h) g[h] = valid ? __ldg(db + ((size_t)h * N + i) * N + j) : 0.f;
    float m1 = 0.f, m2 = 0.f, dxh[PB_C];
#pragma unroll
    for (int c = 0; c < PB_C; ++c) {
      const float xh = (x[c] - mean) * rstd;
      float d = 0.f;
#pragma unroll
      for (int h = 0; h < PB_H; ++h) { d = fmaf(g[h], w[h * PB_C + c], d); acc[h][c] = fmaf(g[h], xh, acc[h][c]); }
      dxh[c] = d; m1 += d; m2 += d * xh; x[c] = xh;
    }
    m1 *= 1.f / PB_C; m2 *= 1.f / PB_C;
    float out[PB_C];
#pragma unroll
    for (int c = 0; c < PB_C; ++c) out[c] = (dxh[c] - m1 - x[c] * m2) * rstd;
    pb_store_row(dz + e * PB_C, out);
  }
  // block partial of d(W'): warp shuffle, then across the 8 warps
#pragma unroll
  for (int h = 0; h < PB_H; ++h)
#pragma unroll
    for (int c = 0; c < PB_C; ++c) {
      float v = acc[h][c];
#pragma unroll
      for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
      if ((tid & 31) == 0) red[tid >> 5][h * PB_C + c] = v;
    }
  __syncthreads();
  if (tid < PB_H * PB_C) {
    float v = 0.f;
#pragma unroll
    for (int r = 0; r < PB_NT / 32; ++r) v += red[r][tid];
    pw[((size_t)blockIdx.y * gridDim.x + blockIdx.x) * PB_H * PB_C + tid] = v;
  }
}

// ---- the pair bias at any pair width: bias[h, i, j] = oscale rstd (z[i, j, :] . W'[h] - mean sum_c W'[h, c]) for C channels (a multiple of 8) and H heads (a multiple of 4)
// One thread per column j and R rows (i0 + ty + 4 r), as ``pair_bias_fwd_kernel``, but STREAMING over the channels in chunks of 8: the row sums, the sums of squares and the H dot products
// (fp32 FMAs against W' held channel-major [C][H] in shared memory, read as warp-uniform float4s, each used by the thread's R rows) accumulate in registers, so a row never has to fit
// in them.  The pair is read once; a block covers 64 columns x 4 R rows.
template <int H> struct PbGenCfg { static_assert(H % 4 == 0 && H <= 32, "heads: a multiple of 4"); };
__device__ __forceinline__ void pb_load8(const __nv_bfloat16* p, float (&x)[8]) {
  const uint4 a = __ldg(reinterpret_cast<const uint4*>(p));
  const uint32_t w[4] = {a.x, a.y, a.z, a.w};
#pragma unroll
  for (int k = 0; k < 4; ++k) { x[2 * k] = __uint_as_float(w[k] << 16); x[2 * k + 1] = __uint_as_float(w[k] & 0xffff0000u); }
}
__device__ __forceinline__ void pb_load8(const float* p, float (&x)[8]) {
  const float4 a = __ldg(reinterpret_cast<const float4*>(p)), b = __ldg(reinterpret_cast<const float4*>(p) + 1);
  x[0] = a.x; x[1] = a.y; x[2] = a.z; x[3] = a.w; x[4] = b.x; x[5] = b.y; x[6] = b.z; x[7] = b.w;
}

__device__ __forceinline__ void pb_store8(__nv_bfloat16* p, const float (&v)[8]) {                  // 8 values as one 16-byte store
  uint32_t o[4];
#pragma unroll
  for (int k = 0; k < 4; ++k) { const __nv_bfloat162 t = __floats2bfloat162_rn(v[2 * k], v[2 * k + 1]); o[k] = *reinterpret_cast<const uint32_t*>(&t); }
  *reinterpret_cast<uint4*>(p) = make_uint4(o[0], o[1], o[2], o[3]);
}
__device__ __forceinline__ void pb_store8(float* p, const float (&v)[8]) {
  float4* q = reinterpret_cast<float4*>(p);
  q[0] = make_float4(v[0], v[1], v[2], v[3]); q[1] = make_float4(v[4], v[5], v[6], v[7]);
}

template <int H, int R, bool PAD, typename TI, typename TO>
__global__ void __launch_bounds__(PB_NT) pair_bias_gen_kernel(const TI* __restrict__ z, const float* __restrict__ wcm, TO* __restrict__ bo, int N, int NZ, int C,
                                                              const unsigned char* __restrict__ kv, float eps, float oscale, float fill) {
  PbGenCfg<H>();
  extern __shared__ __align__(16) float gsm[];                           // W' channel-major [C][H], then sum_c W' [H]
  float* ws = gsm + (size_t)C * H;
  const int tid = threadIdx.x, tx = tid & 63, ty = tid >> 6;
  for (int e = tid; e < C * H; e += PB_NT) gsm[e] = wcm[e];
  __syncthreads();
  if (tid < H) { float s = 0.f; for (int c = 0; c < C; ++c) s += gsm[c * H + tid]; ws[tid] = s; }
  __syncthreads();
  const int i0 = blockIdx.y * (4 * R), j = blockIdx.x * PB_TJ + tx;
  const bool jin = !PAD || j < NZ;
  const bool valid = jin && (kv == nullptr || kv[j]);
  float d[R][H], s[R], s2[R];
  bool live[R];
  const TI* row[R];
#pragma unroll
  for (int r = 0; r < R; ++r) {
    const int i = i0 + ty + 4 * r;
    live[r] = !PAD || (i < NZ && jin);
    row[r] = z + (live[r] ? ((size_t)i * NZ + j) * C : (size_t)0);
    s[r] = s2[r] = 0.f;
#pragma unroll
    for (int h = 0; h < H; ++h) d[r][h] = 0.f;
  }
#pragma unroll 1
  for (int c0 = 0; c0 < C; c0 += 8) {
    float x[R][8];
#pragma unroll
    for (int r = 0; r < R; ++r) {
      if (live[r]) pb_load8(row[r] + c0, x[r]);
      else {
#pragma unroll
        for (int e = 0; e < 8; ++e) x[r][e] = 0.f;
      }
    }
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const float4* wr = reinterpret_cast<const float4*>(gsm + (size_t)(c0 + e) * H);
#pragma unroll
      for (int hq = 0; hq < H / 4; ++hq) {
        const float4 w = wr[hq];
#pragma unroll
        for (int r = 0; r < R; ++r) {
          d[r][4 * hq] = fmaf(x[r][e], w.x, d[r][4 * hq]); d[r][4 * hq + 1] = fmaf(x[r][e], w.y, d[r][4 * hq + 1]);
          d[r][4 * hq + 2] = fmaf(x[r][e], w.z, d[r][4 * hq + 2]); d[r][4 * hq + 3] = fmaf(x[r][e], w.w, d[r][4 * hq + 3]);
        }
      }
#pragma unroll
      for (int r = 0; r < R; ++r) { s[r] += x[r][e]; s2[r] = fmaf(x[r][e], x[r][e], s2[r]); }
    }
  }
  const float invc = 1.f / (float)C;
#pragma unroll
  for (int r = 0; r < R; ++r) {
    const int i = i0 + ty + 4 * r;
    const float mean = s[r] * invc, rstd = rsqrtf(fmaxf(s2[r] * invc - mean * mean, 0.f) + eps);
#pragma unroll
    for (int h = 0; h < H; ++h) {
      const float y = live[r] ? rstd * (d[r][h] - mean * ws[h]) : 0.f;       // a padded query row or key: 0
      pb_store(bo + ((size_t)h * N + i) * N + j, valid ? y * oscale : fill);
    }
  }
}

// ---- the backward of the pair bias at any width: dz = LN_bwd(sum_h db[h] W'[h]) and dW'[h, c] = sum_ij db[h, i, j] xh[i, j, c]  (xh = (z - mean) rstd: LayerNorm without affine; W' folded)
// A block takes RPC rows i of TE consecutive columns j: the tile's z (TE x C) and its dbias (TE x H, a masked key's / a padded element's 0) go to shared memory, then
//   dz phase   TPE = 256 / TE threads per element take interleaved channels (c = q + TPE k: conflict-free with the row pitch C + 4): statistics by warp shuffles, xh in place of z,
//              dxh_c = sum_h db[h] W'[c][h] (FMAs against W' channel-major in shared memory), m1 = mean dxh, m2 = mean dxh xh, dz = rstd (dxh - m1 - xh m2) staged and stored as coalesced vectors;
//   dW phase   thread (head group, channel slot) owns HB = H / 4 heads x CB = C / 64 channels of dW' and runs over the tile's elements and rows: per element one g read per head and one xh read per
//              channel feed HB x CB FMAs; the block's partial [H, C] goes to pw (the caller sums the blocks in a fixed order).
// C a multiple of 64 (64, 128, 256, 512: TE = 128, 64, 32, 16), H = 8, 12, 16 or 24 (a multiple of 4).
template <int C_> struct PbBwdCfg {
  static constexpr int C = C_, TE = C_ == 64 ? 128 : C_ == 128 ? 64 : C_ == 256 ? 32 : 16, TPE = 256 / TE, P = C_ + 4, CB = C_ / 64;
  static_assert(C_ % 64 == 0 && C_ <= 512, "C: 64 .. 512, a multiple of 64");
  static constexpr int smem(int H) { return (C_ * H + 2 * TE * P + TE * H) * 4; }
};

template <int H, int C, typename T>
__global__ void __launch_bounds__(256) pair_bias_gen_bwd_kernel(const T* __restrict__ z, const float* __restrict__ wcm, const float* __restrict__ db, T* __restrict__ dz, float* __restrict__ pw,
                                                                int N, int NZ, int RPC, const unsigned char* __restrict__ kv, float eps) {
  using G = PbBwdCfg<C>;
  constexpr int TE = G::TE, TPE = G::TPE, P = G::P, CB = G::CB, HB = H / 4;
  static_assert(H % 4 == 0, "heads: a multiple of 4");
  extern __shared__ __align__(16) float bsm[];
  float* ws = bsm;                                                       // W' channel-major [C][H]
  float* xh = ws + C * H;                                                // [TE][P]: z, then xh
  float* dx = xh + TE * P;                                               // [TE][P]: dxh, then dz
  float* gs = dx + TE * P;                                               // [TE][H]
  const int tid = threadIdx.x, e = tid / TPE, q = tid % TPE, hg = tid >> 6, slot = tid & 63;
  for (int k = tid; k < C * H; k += 256) ws[k] = wcm[k];
  const int j0 = blockIdx.x * TE;
  float acc[HB][CB];
#pragma unroll
  for (int hb = 0; hb < HB; ++hb)
#pragma unroll
    for (int cb = 0; cb < CB; ++cb) acc[hb][cb] = 0.f;
  for (int r = 0; r < RPC; ++r) {
    const int i = blockIdx.y * RPC + r;
    if (i >= NZ) break;                                                  // uniform over the block
    __syncthreads();                                                     // the previous row's readers are done with xh / dx / gs
    for (int idx = tid; idx < TE * (C / 8); idx += 256) {                // the z tile: 8 channels per task
      const int el = idx / (C / 8), c8 = (idx - el * (C / 8)) * 8;
      float v[8];
      if (j0 + el < NZ) pb_load8(z + ((size_t)i * NZ + j0 + el) * C + c8, v);
      else {
#pragma unroll
        for (int k = 0; k < 8; ++k) v[k] = 0.f;
      }
      float4* d = reinterpret_cast<float4*>(xh + el * P + c8);
      d[0] = make_float4(v[0], v[1], v[2], v[3]); d[1] = make_float4(v[4], v[5], v[6], v[7]);
    }
    for (int idx = tid; idx < TE * H; idx += 256) {                      // g[el][h]: a masked key and a padded element have none
      const int h = idx / TE, el = idx - h * TE, j = j0 + el;
      gs[el * H + h] = (j < NZ && (kv == nullptr || kv[j])) ? __ldg(db + ((size_t)h * N + i) * N + j) : 0.f;
    }
    __syncthreads();
    // ---- dz phase: element e, channels q, q + TPE, ...
    float* xr = xh + e * P;
    float* dr = dx + e * P;
    float s = 0.f, s2 = 0.f;
#pragma unroll 8
    for (int k = 0; k < C / TPE; ++k) { const float x = xr[q + TPE * k]; s += x; s2 = fmaf(x, x, s2); }
#pragma unroll
    for (int o = TPE / 2; o > 0; o >>= 1) { s += __shfl_xor_sync(0xffffffffu, s, o); s2 += __shfl_xor_sync(0xffffffffu, s2, o); }
    const float mean = s * (1.f / C), rstd = rsqrtf(fmaxf(s2 * (1.f / C) - mean * mean, 0.f) + eps);
    float g[H];
#pragma unroll
    for (int h = 0; h < H; ++h) g[h] = gs[e * H + h];
    float m1 = 0.f, m2 = 0.f;
#pragma unroll 4
    for (int k = 0; k < C / TPE; ++k) {
      const int c = q + TPE * k;
      const float xn = (xr[c] - mean) * rstd;
      xr[c] = xn;
      const float4* wr = reinterpret_cast<const float4*>(ws + (size_t)c * H);
      float d = 0.f;
#pragma unroll
      for (int hq = 0; hq < H / 4; ++hq) {
        const float4 w = wr[hq];
        d = fmaf(g[4 * hq], w.x, d); d = fmaf(g[4 * hq + 1], w.y, d); d = fmaf(g[4 * hq + 2], w.z, d); d = fmaf(g[4 * hq + 3], w.w, d);
      }
      dr[c] = d; m1 += d; m2 = fmaf(d, xn, m2);
    }
#pragma unroll
    for (int o = TPE / 2; o > 0; o >>= 1) { m1 += __shfl_xor_sync(0xffffffffu, m1, o); m2 += __shfl_xor_sync(0xffffffffu, m2, o); }
    m1 *= 1.f / C; m2 *= 1.f / C;
#pragma unroll 4
    for (int k = 0; k < C / TPE; ++k) { const int c = q + TPE * k; dr[c] = rstd * (dr[c] - m1 - xr[c] * m2); }
    __syncthreads();                                                     // dz and xh of every element are complete
    for (int idx = tid; idx < TE * (C / 8); idx += 256) {                // dz: coalesced 16-byte stores along the tile
      const int el = idx / (C / 8), c8 = (idx - el * (C / 8)) * 8;
      if (j0 + el >= NZ) continue;
      const float4* s4 = reinterpret_cast<const float4*>(dx + el * P + c8);
      const float4 a = s4[0], b = s4[1];
      const float v[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
      pb_store8(dz + ((size_t)i * NZ + j0 + el) * C + c8, v);
    }
    // ---- dW phase: this thread's HB heads x CB channels over the tile's elements
#pragma unroll 4
    for (int t = 0; t < TE; ++t) {
      float gv[HB], xv[CB];
#pragma unroll
      for (int hb = 0; hb < HB; ++hb) gv[hb] = gs[t * H + hg * HB + hb];
#pragma unroll
      for (int cb = 0; cb < CB; ++cb) xv[cb] = xh[t * P + slot + 64 * cb];
#pragma unroll
      for (int hb = 0; hb < HB; ++hb)
#pragma unroll
        for (int cb = 0; cb < CB; ++cb) acc[hb][cb] = fmaf(gv[hb], xv[cb], acc[hb][cb]);
    }
  }
  float* out = pw + ((size_t)blockIdx.y * gridDim.x + blockIdx.x) * H * C;
#pragma unroll
  for (int hb = 0; hb < HB; ++hb)
#pragma unroll
    for (int cb = 0; cb < CB; ++cb) out[(size_t)(hg * HB + hb) * C + slot + 64 * cb] = acc[hb][cb];
}

// delta[a][h][i] = sum_d dO o: one thread per (token, head); the thread's 16-byte vectors cover its head's HD values.
template <int HD, typename OT>
__global__ void __launch_bounds__(128) attn_delta_kernel(const __nv_bfloat16* __restrict__ dO, const OT* __restrict__ O, float* __restrict__ delta,
                                                         long long lddo, long long ldo, int L, int H, long long tokens) {
  const long long t = (long long)blockIdx.x * 128 + threadIdx.x;
  const int h = blockIdx.y;
  if (t >= tokens) return;
  const __nv_bfloat16* d = dO + t * lddo + h * HD;
  const OT* o = O + t * ldo + h * HD;
  float s = 0.f;
#pragma unroll
  for (int c = 0; c < HD / 8; ++c) {
    const uint4 u = __ldg(reinterpret_cast<const uint4*>(d) + c);
    const uint32_t w[4] = {u.x, u.y, u.z, u.w};
    float dv[8];
#pragma unroll
    for (int k = 0; k < 4; ++k) { dv[2 * k] = __uint_as_float(w[k] << 16); dv[2 * k + 1] = __uint_as_float(w[k] & 0xffff0000u); }
    float ov[8];
    if constexpr (sizeof(OT) == 2) {
      const uint4 v = __ldg(reinterpret_cast<const uint4*>(o) + c);
      const uint32_t x[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
      for (int k = 0; k < 4; ++k) { ov[2 * k] = __uint_as_float(x[k] << 16); ov[2 * k + 1] = __uint_as_float(x[k] & 0xffff0000u); }
    } else {
      const float4 a = __ldg(reinterpret_cast<const float4*>(o) + 2 * c), b = __ldg(reinterpret_cast<const float4*>(o) + 2 * c + 1);
      ov[0] = a.x; ov[1] = a.y; ov[2] = a.z; ov[3] = a.w; ov[4] = b.x; ov[5] = b.y; ov[6] = b.z; ov[7] = b.w;
    }
#pragma unroll
    for (int k = 0; k < 8; ++k) s = fmaf(dv[k], ov[k], s);
  }
  const long long a = t / L, i = t % L;
  delta[(a * H + h) * L + i] = s;
}

// ---- pair_bias_gen_bwd_tc: the fp32 backward of pair_bias_gen on tensor cores, batched (the AttentionPairBias composed path's wide pair, d_pair 128 / 256 / 384 / 512)
// The same math as pair_bias_gen_bwd -- dz = rstd (dxh - mean(dxh) - xh mean(dxh xh)), dxh = db W'^T, dW' = db^T xh (xh = (z - mean) rstd) -- with both products on TF32
// mma.sync m16n8k8 (fp32 accumulation; dxh in 3xTF32 -- each operand split into two TF32 parts, ~fp32 accurate --, dW' in 1xTF32 with the operands rounded to the
// nearest, as a TF32 GEMM) and one read of z / one write of dz:
//   a CTA of 8 warps is persistent over tiles of 16 consecutive pair elements (i, j) of one batch row; each warp owns C / 8 channels.  A tile's z (16 x C) and its
//   dbias (16 x H, a masked key's / a tail element's 0) arrive in shared memory through a cp.async ring (2 - 3 stages); the warp keeps its z slice in the mma accumulator
//   layout in registers: two-pass row statistics (mean, then the centred variance) by quad shuffles + a cross-warp shared-memory sum in warp order,
//   dxh = g (16 x H) W'^T (H x C/8) on tensor cores (3xTF32), m1 / m2 likewise, dz stored straight from the accumulator registers;
//   dW'^T (C/8 x H per warp) += xh^T (C/8 x 16) g (16 x H) on tensor cores (xh re-read from the shared tile in the A-fragment layout), kept in registers over the CTA's tiles.
// The CTA's dW' partial [H, C] goes to pw[b][cta]; pair_bias_pw_reduce sums them per batch row in CTA order (fixed: a replay is bit-identical, and a batch row's dW' does not
// depend on B -- the per-row CTA count is fixed by the caller).  C a multiple of 128 (<= 512), H a multiple of 8 (<= 24).
DEVI uint32_t pbt_tf32(float x) {
  uint32_t r;
  asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(x));
  return r;
}
// x = hi + lo (both TF32): the 3xTF32 split (hi lo + lo hi + hi hi keeps ~fp32 accuracy)
DEVI void pbt_split(float x, uint32_t& hi, uint32_t& lo) {
  hi = pbt_tf32(x);
  lo = pbt_tf32(x - __uint_as_float(hi));
}
DEVI void pbt_mma(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
DEVI void pbt_cp16(float* dst, const float* src, bool valid) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"((uint32_t)__cvta_generic_to_shared(dst)), "l"(src), "r"(valid ? 16u : 0u) : "memory");
}
DEVI void pbt_cp4(float* dst, const float* src, bool valid) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" ::"r"((uint32_t)__cvta_generic_to_shared(dst)), "l"(src), "r"(valid ? 4u : 0u) : "memory");
}
DEVI void pbt_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
template <int N> DEVI void pbt_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory"); }

template <int C_, int H_> struct PbTcCfg {
  static constexpr int C = C_, H = H_, NW = 8, NT = 256, TM = 16, CW = C_ / NW, NN = CW / 8, MT = CW / 16, KH = H_ / 8;
  static constexpr int P = C_ + 8;          // z row pitch: the accumulator-layout float2 reads and the A-fragment reads are conflict-free (P = 8 mod 32)
  static constexpr int HP = H_ + 4;         // dbias / W' row pitch (H + 4 = 4 mod 8: the fragment reads of 8 rows x 4 heads are conflict-free)
  static constexpr int ZST = TM * P, GST = TM * HP, RED = 4 * NW * TM + 2 * TM;
  static constexpr int FIXED = C_ * HP + RED;
  static constexpr int STAGES = (FIXED + 3 * (ZST + GST)) * 4 <= 160 * 1024 ? 3 : 2;
  static constexpr int SMEM = (FIXED + STAGES * (ZST + GST)) * 4;
  static_assert(C_ % 128 == 0 && C_ <= 512 && H_ % 8 == 0 && H_ <= 24, "C: 128 .. 512 (a multiple of 128), H: 8, 16 or 24");
};

template <int C, int H>
__global__ void __launch_bounds__(256, 1) pair_bias_gen_bwd_tc_kernel(const float* __restrict__ z, const float* __restrict__ wcm, const float* __restrict__ db, float* __restrict__ dz,
                                                                      float* __restrict__ pw, int N, int NZ, const unsigned char* __restrict__ kv, float eps) {
  using G = PbTcCfg<C, H>;
  constexpr int TM = G::TM, P = G::P, HP = G::HP, NW = G::NW, CW = G::CW, NN = G::NN, MT = G::MT, KH = G::KH, ST = G::STAGES;
  extern __shared__ __align__(16) float tsm[];
  float* ws = tsm;                                  // W' channel-major [C][HP], fp32
  float* red = ws + C * HP;                         // [4][NW][TM]: row sums, centred squares, m1, m2 per warp
  float* stat = red + 4 * NW * TM;                  // [TM][2]: mean, rstd of the tile's rows
  float* zst = stat + 2 * TM;                       // [ST][TM][P]
  float* gst = zst + ST * G::ZST;                   // [ST][TM][HP]
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, gq = lane >> 2, tq = lane & 3, c0 = warp * CW;
  const int b = blockIdx.y, nct = gridDim.x;
  const long long R = (long long)NZ * NZ;
  const long long tiles = (R + TM - 1) / TM;
  z += (size_t)b * R * C;
  dz += (size_t)b * R * C;
  db += (size_t)b * H * N * N;
  if (kv != nullptr) kv += (size_t)b * NZ;
  for (int k = tid; k < C * H; k += G::NT) ws[(k / H) * HP + k % H] = wcm[k];

  auto load = [&](long long t, int s) {
    const long long r0 = t * TM;
    float* zs = zst + s * G::ZST;
    float* gs = gst + s * G::GST;
#pragma unroll
    for (int idx = tid; idx < TM * (C / 4); idx += G::NT) {
      const int row = idx / (C / 4), c4 = (idx - row * (C / 4)) * 4;
      const bool ok = r0 + row < R;
      pbt_cp16(zs + row * P + c4, ok ? z + (size_t)(r0 + row) * C + c4 : z, ok);
    }
    for (int idx = tid; idx < TM * H; idx += G::NT) {
      const int h = idx / TM, row = idx - h * TM;
      const long long r = r0 + row;
      const int i = (int)(r / NZ), j = (int)(r - (long long)i * NZ);
      const bool ok = r < R && (kv == nullptr || kv[j]);
      pbt_cp4(gs + row * HP + h, ok ? db + ((size_t)h * N + i) * N + j : db, ok);
    }
  };

  float acc[MT][KH][4];
#pragma unroll
  for (int m = 0; m < MT; ++m)
#pragma unroll
    for (int n = 0; n < KH; ++n)
#pragma unroll
      for (int e = 0; e < 4; ++e) acc[m][n][e] = 0.f;

#pragma unroll
  for (int s = 0; s < ST - 1; ++s) {
    const long long t = blockIdx.x + (long long)s * nct;
    if (t < tiles) load(t, s);
    pbt_commit();
  }
  const float invc = 1.f / (float)C;
  int it = 0;
  for (long long t = blockIdx.x; t < tiles; t += nct, ++it) {
    const int s = it % ST;
    pbt_wait<ST - 2>();
    __syncthreads();                                // stage s has landed for every thread; every warp is done with the stage the next load overwrites
    {
      const long long tn = t + (long long)(ST - 1) * nct;
      if (tn < tiles) load(tn, (it + ST - 1) % ST);
      pbt_commit();
    }
    const float* zs = zst + s * G::ZST;
    const float* gs = gst + s * G::GST;
    const long long r0 = t * TM;
    // z of rows gq, gq + 8 at the accumulator columns 8 n + 2 tq, + 1 of this warp's channels
    float x[NN][4];
#pragma unroll
    for (int n = 0; n < NN; ++n) {
      const float2 u = *reinterpret_cast<const float2*>(zs + gq * P + c0 + 8 * n + 2 * tq);
      const float2 v = *reinterpret_cast<const float2*>(zs + (gq + 8) * P + c0 + 8 * n + 2 * tq);
      x[n][0] = u.x; x[n][1] = u.y; x[n][2] = v.x; x[n][3] = v.y;
    }
    float s0 = 0.f, s1 = 0.f;
#pragma unroll
    for (int n = 0; n < NN; ++n) { s0 += x[n][0] + x[n][1]; s1 += x[n][2] + x[n][3]; }
    s0 += __shfl_xor_sync(0xffffffffu, s0, 1); s0 += __shfl_xor_sync(0xffffffffu, s0, 2);
    s1 += __shfl_xor_sync(0xffffffffu, s1, 1); s1 += __shfl_xor_sync(0xffffffffu, s1, 2);
    if (tq == 0) { red[warp * TM + gq] = s0; red[warp * TM + gq + 8] = s1; }
    __syncthreads();
    float mean0 = 0.f, mean1 = 0.f;
#pragma unroll
    for (int w = 0; w < NW; ++w) { mean0 += red[w * TM + gq]; mean1 += red[w * TM + gq + 8]; }
    mean0 *= invc; mean1 *= invc;
    float v0 = 0.f, v1 = 0.f;
#pragma unroll
    for (int n = 0; n < NN; ++n) {
      x[n][0] -= mean0; x[n][1] -= mean0; x[n][2] -= mean1; x[n][3] -= mean1;
      v0 = fmaf(x[n][0], x[n][0], fmaf(x[n][1], x[n][1], v0));
      v1 = fmaf(x[n][2], x[n][2], fmaf(x[n][3], x[n][3], v1));
    }
    v0 += __shfl_xor_sync(0xffffffffu, v0, 1); v0 += __shfl_xor_sync(0xffffffffu, v0, 2);
    v1 += __shfl_xor_sync(0xffffffffu, v1, 1); v1 += __shfl_xor_sync(0xffffffffu, v1, 2);
    if (tq == 0) { red[(NW + warp) * TM + gq] = v0; red[(NW + warp) * TM + gq + 8] = v1; }
    __syncthreads();
    float var0 = 0.f, var1 = 0.f;
#pragma unroll
    for (int w = 0; w < NW; ++w) { var0 += red[(NW + w) * TM + gq]; var1 += red[(NW + w) * TM + gq + 8]; }
    const float rstd0 = rsqrtf(var0 * invc + eps), rstd1 = rsqrtf(var1 * invc + eps);
    if (warp == 0 && tq == 0) { stat[2 * gq] = mean0; stat[2 * gq + 1] = rstd0; stat[2 * (gq + 8)] = mean1; stat[2 * (gq + 8) + 1] = rstd1; }
    // dxh = g W'^T: A = g [16 rows x H], B = W'^T [H x 8 channels]
    float d[NN][4];
#pragma unroll
    for (int n = 0; n < NN; ++n)
#pragma unroll
      for (int e = 0; e < 4; ++e) d[n][e] = 0.f;
#pragma unroll
    for (int kk = 0; kk < KH; ++kk) {
      uint32_t ah[4], al[4];
      pbt_split(gs[gq * HP + 8 * kk + tq], ah[0], al[0]);
      pbt_split(gs[(gq + 8) * HP + 8 * kk + tq], ah[1], al[1]);
      pbt_split(gs[gq * HP + 8 * kk + tq + 4], ah[2], al[2]);
      pbt_split(gs[(gq + 8) * HP + 8 * kk + tq + 4], ah[3], al[3]);
#pragma unroll
      for (int n = 0; n < NN; ++n) {
        const float* wr = ws + (c0 + 8 * n + gq) * HP + 8 * kk + tq;
        uint32_t bh0, bl0, bh1, bl1;
        pbt_split(wr[0], bh0, bl0);
        pbt_split(wr[4], bh1, bl1);
        pbt_mma(d[n], al[0], al[1], al[2], al[3], bh0, bh1);   // 3xTF32: the small cross terms first, then hi x hi
        pbt_mma(d[n], ah[0], ah[1], ah[2], ah[3], bl0, bl1);
        pbt_mma(d[n], ah[0], ah[1], ah[2], ah[3], bh0, bh1);
      }
    }
    float m10 = 0.f, m11 = 0.f, m20 = 0.f, m21 = 0.f;
#pragma unroll
    for (int n = 0; n < NN; ++n) {
      x[n][0] *= rstd0; x[n][1] *= rstd0; x[n][2] *= rstd1; x[n][3] *= rstd1;   // xh
      m10 += d[n][0] + d[n][1]; m11 += d[n][2] + d[n][3];
      m20 = fmaf(d[n][0], x[n][0], fmaf(d[n][1], x[n][1], m20));
      m21 = fmaf(d[n][2], x[n][2], fmaf(d[n][3], x[n][3], m21));
    }
#pragma unroll
    for (int o = 1; o <= 2; o <<= 1) {
      m10 += __shfl_xor_sync(0xffffffffu, m10, o); m11 += __shfl_xor_sync(0xffffffffu, m11, o);
      m20 += __shfl_xor_sync(0xffffffffu, m20, o); m21 += __shfl_xor_sync(0xffffffffu, m21, o);
    }
    if (tq == 0) {
      red[(2 * NW + warp) * TM + gq] = m10; red[(2 * NW + warp) * TM + gq + 8] = m11;
      red[(3 * NW + warp) * TM + gq] = m20; red[(3 * NW + warp) * TM + gq + 8] = m21;
    }
    __syncthreads();                                // m1 / m2 partials and the rows' statistics are in shared memory
    float M10 = 0.f, M11 = 0.f, M20 = 0.f, M21 = 0.f;
#pragma unroll
    for (int w = 0; w < NW; ++w) {
      M10 += red[(2 * NW + w) * TM + gq]; M11 += red[(2 * NW + w) * TM + gq + 8];
      M20 += red[(3 * NW + w) * TM + gq]; M21 += red[(3 * NW + w) * TM + gq + 8];
    }
    M10 *= invc; M11 *= invc; M20 *= invc; M21 *= invc;
    const bool live0 = r0 + gq < R, live1 = r0 + gq + 8 < R;
    float* o0 = dz + (size_t)(r0 + gq) * C + c0 + 2 * tq;
    float* o1 = dz + (size_t)(r0 + gq + 8) * C + c0 + 2 * tq;
#pragma unroll
    for (int n = 0; n < NN; ++n) {
      if (live0) *reinterpret_cast<float2*>(o0 + 8 * n) = make_float2(rstd0 * (d[n][0] - M10 - x[n][0] * M20), rstd0 * (d[n][1] - M10 - x[n][1] * M20));
      if (live1) *reinterpret_cast<float2*>(o1 + 8 * n) = make_float2(rstd1 * (d[n][2] - M11 - x[n][2] * M21), rstd1 * (d[n][3] - M11 - x[n][3] * M21));
    }
    // dW'^T += xh^T g: A = xh^T [16 channels x 8 rows], B = g [8 rows x 8 heads]
#pragma unroll
    for (int ks = 0; ks < 2; ++ks) {
      const int ra = 8 * ks + tq, rb = ra + 4;
      const float ma = stat[2 * ra], sa = stat[2 * ra + 1], mb = stat[2 * rb], sb = stat[2 * rb + 1];
      uint32_t b0[KH], b1[KH];
#pragma unroll
      for (int n = 0; n < KH; ++n) { b0[n] = pbt_tf32(gs[ra * HP + 8 * n + gq]); b1[n] = pbt_tf32(gs[rb * HP + 8 * n + gq]); }
#pragma unroll
      for (int m = 0; m < MT; ++m) {
        const int ch = c0 + 16 * m + gq;
        const uint32_t a0 = pbt_tf32((zs[ra * P + ch] - ma) * sa), a1 = pbt_tf32((zs[ra * P + ch + 8] - ma) * sa);
        const uint32_t a2 = pbt_tf32((zs[rb * P + ch] - mb) * sb), a3 = pbt_tf32((zs[rb * P + ch + 8] - mb) * sb);
#pragma unroll
        for (int n = 0; n < KH; ++n) pbt_mma(acc[m][n], a0, a1, a2, a3, b0[n], b1[n]);
      }
    }
  }
  pbt_wait<0>();
  float* out = pw + ((size_t)b * nct + blockIdx.x) * H * C;
#pragma unroll
  for (int m = 0; m < MT; ++m)
#pragma unroll
    for (int n = 0; n < KH; ++n) {
      const int ch = c0 + 16 * m + gq, h = 8 * n + 2 * tq;
      out[(size_t)h * C + ch] = acc[m][n][0];
      out[(size_t)(h + 1) * C + ch] = acc[m][n][1];
      out[(size_t)h * C + ch + 8] = acc[m][n][2];
      out[(size_t)(h + 1) * C + ch + 8] = acc[m][n][3];
    }
}

// dw[b][e] = sum over the CTAs g = 0, 1, ... (in order) of pw[b][g][e]: the per-batch-row fixed-order sum of pair_bias_gen_bwd_tc's partials
__global__ void __launch_bounds__(256) pair_bias_pw_reduce_kernel(const float* __restrict__ pw, float* __restrict__ dw, int nct, int HC) {
  const int e = blockIdx.x * 256 + threadIdx.x, b = blockIdx.y;
  if (e >= HC) return;
  const float* p = pw + (size_t)b * nct * HC + e;
  float s = 0.f;
  for (int g = 0; g < nct; ++g) s += p[(size_t)g * HC];
  dw[(size_t)b * HC + e] = s;
}

}  // namespace aa80
