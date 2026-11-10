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

}  // namespace aa80
