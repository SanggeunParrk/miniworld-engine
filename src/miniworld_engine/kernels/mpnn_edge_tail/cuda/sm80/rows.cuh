// rows.cuh -- the two streaming kernels of the memory policies on the A100 (sm_80): the edge LayerNorm's backward from the compressed (bf16) input copy and the edge dropout's bit-packed mask
// (pack the ATen mask to one bit per element; the backward that reads the bits).  Both are bandwidth-bound: 16-byte vectors per thread, several rows in flight per warp.
#pragma once
#include "common.cuh"

namespace me80 {

// ---------------------------------------------------------------------------------------------------------------------------------------------- LayerNorm backward
// dx = rstd (w dy - (xhat mean(w dy xhat) + mean(w dy))),  dw = sum dy xhat,  db = sum dy; width 128, a half warp owns a row (lane l of 16: channels 8 l .. 8 l + 7, one 16-byte load of the
// bf16 input and of a bf16 gradient), a warp takes two rows at a time, 8 warps per CTA, 4 row pairs of every warp in flight.
struct LnBwdParams {
  const void* dy;              // [rows][128] bf16 or fp32
  const __nv_bfloat16* x;      // [rows][128] the forward's input rounded to bf16
  const float* mean;           // [rows]
  const float* rstd;           // [rows]
  const void* w;               // [128] bf16 or fp32
  void* dx;                    // [rows][128] in dy's dtype
  float* part;                 // [grid][2][128] per-CTA partial sums of dw | db
  int rows, dy_fp32, w_fp32;
};

DEVI void ln_load8(const void* p, int fp32, float (&v)[8]) {
  if (fp32) {
    const float4 a = *reinterpret_cast<const float4*>(p), b = *(reinterpret_cast<const float4*>(p) + 1);
    v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w; v[4] = b.x; v[5] = b.y; v[6] = b.z; v[7] = b.w;
  } else {
    const uint4 r = *reinterpret_cast<const uint4*>(p);
    v[0] = bf16lo(r.x); v[1] = bf16hi(r.x); v[2] = bf16lo(r.y); v[3] = bf16hi(r.y); v[4] = bf16lo(r.z); v[5] = bf16hi(r.z); v[6] = bf16lo(r.w); v[7] = bf16hi(r.w);
  }
}
DEVI void ln_store8(void* p, int fp32, const float (&v)[8]) {
  if (fp32) {
    *reinterpret_cast<float4*>(p) = make_float4(v[0], v[1], v[2], v[3]);
    *(reinterpret_cast<float4*>(p) + 1) = make_float4(v[4], v[5], v[6], v[7]);
  } else {
    *reinterpret_cast<uint4*>(p) = make_uint4(pack_bf16(v[0], v[1]), pack_bf16(v[2], v[3]), pack_bf16(v[4], v[5]), pack_bf16(v[6], v[7]));
  }
}
// sum over the 16 lanes of a half warp
DEVI float half_sum(float v) {
#pragma unroll
  for (int o = 8; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

__global__ void __launch_bounds__(256) ln_bwd_kernel(const LnBwdParams p) {
  __shared__ float red[8][256];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, hlf = lane >> 4, l16 = lane & 15;
  float wv[8];
  ln_load8(reinterpret_cast<const char*>(p.w) + (size_t)l16 * 8 * (p.w_fp32 ? 4 : 2), p.w_fp32, wv);
  float dw[8], db[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) { dw[i] = 0.f; db[i] = 0.f; }
  const size_t esz = p.dy_fp32 ? 4 : 2;
  const int stride = gridDim.x * 8;                                  // row pairs advanced by one pass of the whole grid
  for (int q0 = blockIdx.x * 8 + warp; 2 * q0 < p.rows; q0 += 4 * stride) {
    float xh[4][8], dyv[4][8], rs[4];
    bool v[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const int r = 2 * (q0 + k * stride) + hlf;
      v[k] = r < p.rows;
      const int rr = v[k] ? r : p.rows - 1;
      const uint4 xr = *reinterpret_cast<const uint4*>(p.x + (size_t)rr * D + 8 * l16);
      const float mean = p.mean[rr];
      rs[k] = p.rstd[rr];
      xh[k][0] = (bf16lo(xr.x) - mean) * rs[k]; xh[k][1] = (bf16hi(xr.x) - mean) * rs[k]; xh[k][2] = (bf16lo(xr.y) - mean) * rs[k]; xh[k][3] = (bf16hi(xr.y) - mean) * rs[k];
      xh[k][4] = (bf16lo(xr.z) - mean) * rs[k]; xh[k][5] = (bf16hi(xr.z) - mean) * rs[k]; xh[k][6] = (bf16lo(xr.w) - mean) * rs[k]; xh[k][7] = (bf16hi(xr.w) - mean) * rs[k];
      ln_load8(reinterpret_cast<const char*>(p.dy) + ((size_t)rr * D + 8 * l16) * esz, p.dy_fp32, dyv[k]);
    }
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      float s1 = 0.f, s2 = 0.f;
#pragma unroll
      for (int i = 0; i < 8; ++i) { const float t = dyv[k][i] * wv[i]; s1 = fmaf(t, xh[k][i], s1); s2 += t; }
      const float c1 = half_sum(s1) * (1.f / 128.f), c2 = half_sum(s2) * (1.f / 128.f);
      float o[8];
#pragma unroll
      for (int i = 0; i < 8; ++i) o[i] = (dyv[k][i] * wv[i] - fmaf(xh[k][i], c1, c2)) * rs[k];
      if (v[k]) {
        ln_store8(reinterpret_cast<char*>(p.dx) + ((size_t)(2 * (q0 + k * stride) + hlf) * D + 8 * l16) * esz, p.dy_fp32, o);
#pragma unroll
        for (int i = 0; i < 8; ++i) { dw[i] = fmaf(dyv[k][i], xh[k][i], dw[i]); db[i] += dyv[k][i]; }
      }
    }
  }
  // the two half warps hold the same 128 channels: add them, then the 8 warps in order
#pragma unroll
  for (int i = 0; i < 8; ++i) { dw[i] += __shfl_xor_sync(0xffffffffu, dw[i], 16); db[i] += __shfl_xor_sync(0xffffffffu, db[i], 16); }
  if (hlf == 0) {
#pragma unroll
    for (int i = 0; i < 8; ++i) { red[warp][8 * l16 + i] = dw[i]; red[warp][128 + 8 * l16 + i] = db[i]; }
  }
  __syncthreads();
  {
    float s = 0.f;
#pragma unroll
    for (int w = 0; w < 8; ++w) s += red[w][threadIdx.x];
    p.part[(size_t)blockIdx.x * 256 + threadIdx.x] = s;
  }
}

// 8 blocks of 1024 threads (one SM would take ~5 us to pull the 400 KB of partials through its L2 port): block b owns 32 of the 256 columns (dw | db), a warp = 32 consecutive columns of one row
// group, 32 row groups; group g adds its rows (r = g, g + 32, ...) in order and the 32 group sums are added in order: a fixed summation order
__global__ void __launch_bounds__(1024) ln_finalize_kernel(const float* __restrict__ part, int nblocks, void* dw, void* db, int w_fp32) {
  __shared__ float red[32][33];
  const int c = threadIdx.x & 31, g = threadIdx.x >> 5, col = blockIdx.x * 32 + c;       // col: 0..255
  float s = 0.f;
#pragma unroll 8
  for (int r = g; r < nblocks; r += 32) s += part[(size_t)r * 256 + col];
  red[g][c] = s;
  __syncthreads();
  if (g == 0) {
    float t = 0.f;
#pragma unroll
    for (int k = 0; k < 32; ++k) t += red[k][c];
    void* dst = col < 128 ? dw : db;
    if (w_fp32) reinterpret_cast<float*>(dst)[col & 127] = t;
    else reinterpret_cast<__nv_bfloat16*>(dst)[col & 127] = __float2bfloat16_rn(t);
  }
}

// --------------------------------------------------------------------------------------------------------------------------------------------------- dropout
// packed[i / 8] bit (i % 8) = mask[i] (bool, one byte per element); the backward reads the bits.  The element counts are arbitrary (the tails go element by element).
__global__ void __launch_bounds__(256) pack_mask_kernel(const uint8_t* __restrict__ mask, uint8_t* __restrict__ packed, long long n) {
  const long long b = (long long)blockIdx.x * blockDim.x + threadIdx.x;       // output byte
  const long long nbytes = (n + 7) / 8;
  if (b >= nbytes) return;
  const long long i0 = b * 8;
  uint32_t bits = 0;
  if (i0 + 8 <= n) {
    const uint2 m = *reinterpret_cast<const uint2*>(mask + i0);               // i0 is a multiple of 8: 8-byte aligned when the tensor is
    bits = ((m.x & 0x01u) ? 1u : 0u) | ((m.x & 0x0100u) ? 2u : 0u) | ((m.x & 0x010000u) ? 4u : 0u) | ((m.x & 0x01000000u) ? 8u : 0u) |
           ((m.y & 0x01u) ? 16u : 0u) | ((m.y & 0x0100u) ? 32u : 0u) | ((m.y & 0x010000u) ? 64u : 0u) | ((m.y & 0x01000000u) ? 128u : 0u);
  } else {
    for (int j = 0; j < 8 && i0 + j < n; ++j) bits |= (mask[i0 + j] ? 1u : 0u) << j;
  }
  packed[b] = (uint8_t)bits;
}

template <bool FP32>
__global__ void __launch_bounds__(256) dropout_bwd_kernel(const void* __restrict__ grad, const uint8_t* __restrict__ packed, void* __restrict__ out, long long n, float scale) {
  const long long b = (long long)blockIdx.x * blockDim.x + threadIdx.x;       // packed byte = 8 elements
  const long long i0 = b * 8;
  if (i0 >= n) return;
  const uint32_t bits = packed[b];
  if (i0 + 8 <= n) {
    float g[8];
    if (FP32) {
      const float4 a = *reinterpret_cast<const float4*>(reinterpret_cast<const float*>(grad) + i0), c = *(reinterpret_cast<const float4*>(reinterpret_cast<const float*>(grad) + i0) + 1);
      g[0] = a.x; g[1] = a.y; g[2] = a.z; g[3] = a.w; g[4] = c.x; g[5] = c.y; g[6] = c.z; g[7] = c.w;
    } else {
      const uint4 r = *reinterpret_cast<const uint4*>(reinterpret_cast<const __nv_bfloat16*>(grad) + i0);
      g[0] = bf16lo(r.x); g[1] = bf16hi(r.x); g[2] = bf16lo(r.y); g[3] = bf16hi(r.y); g[4] = bf16lo(r.z); g[5] = bf16hi(r.z); g[6] = bf16lo(r.w); g[7] = bf16hi(r.w);
    }
    float o[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) o[j] = ((bits >> j) & 1u) ? g[j] * scale : 0.f;
    if (FP32) {
      float* dst = reinterpret_cast<float*>(out) + i0;
      *reinterpret_cast<float4*>(dst) = make_float4(o[0], o[1], o[2], o[3]);
      *(reinterpret_cast<float4*>(dst) + 1) = make_float4(o[4], o[5], o[6], o[7]);
    } else {
      *reinterpret_cast<uint4*>(reinterpret_cast<__nv_bfloat16*>(out) + i0) =
          make_uint4(pack_bf16(o[0], o[1]), pack_bf16(o[2], o[3]), pack_bf16(o[4], o[5]), pack_bf16(o[6], o[7]));
    }
  } else {
    for (int j = 0; j < 8 && i0 + j < n; ++j) {
      const float g = FP32 ? reinterpret_cast<const float*>(grad)[i0 + j] : __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(grad)[i0 + j]);
      const float o = ((bits >> j) & 1u) ? g * scale : 0.f;
      if (FP32) reinterpret_cast<float*>(out)[i0 + j] = o;
      else reinterpret_cast<__nv_bfloat16*>(out)[i0 + j] = __float2bfloat16_rn(o);
    }
  }
}

}  // namespace me80
