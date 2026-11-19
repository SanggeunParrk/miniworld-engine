// Bias-only token DiT: the row kernel of the HOISTED pair bias's backward (kernels/integrations/bias_only_dit_hoist.py).
//
// The hoist makes every block's attention logits from ONE LayerNorm of the pair rows without affine (LN0: the token DiT rows'
// layernorm128_rows) and ONE cuBLAS GEMM against all blocks' gamma-folded weights. Its backward is two cuBLAS GEMMs
// (d LN0 = dbias_all^T W'_all with fp32 output; dW'_all = dbias_all Y) and this kernel between them:
//
//   ln0_bwd_rows   dx = rstd (dy - mean(dy) - y mean(dy y)),   y = (x - mean) rstd          (128 channels, no affine)
//                  and Y, the dW' GEMM's operand made from the same fp32 y:
//                    bf16 path  Y = [hi | lo] [R, 256] bf16, hi = bf16(y), lo = bf16(y - hi)   (hi + lo = y to ~2^-16 relative)
//                    fp32 path  Y = y [R, 128] fp32 (the TF32 GEMM's operand)
//
// Why the split: dW'_b = dbias_b LN0 feeds d gamma_b = sum_h dW'_b o W_b. With LN0 rounded to bf16 for the GEMM, every product
// carries LN0's rounding (2^-9) -- an error the PyTorch block does not have in d gamma (its LayerNorm backward reads LN0 in fp32).
// The two-term operand makes the GEMM's sum of the bf16 dbias against y exact to fp32 accumulation; the GEMM reads K = 256 columns
// instead of 128. LN0 is no longer kept from the forward: this pass recomputes it.
//
// x the pair rows as the forward read them (bf16 or fp32), dy = d LN0 (fp32), dx in its own dtype (bf16 or fp32). The row
// statistics are recomputed from x with the forward kernel's arithmetic (layernorm_rows_kernel<128>: the same sums in the same
// order), so y is the forward's fp32 value.
//
// SYNC PROTOCOL. None beyond the warp: one warp per RPW consecutive rows (all of their loads issued before any math); a lane owns
// channels 4 lane .. 4 lane + 3 of each row (and the matching columns of both halves of Y); the four row sums (mean, variance,
// mean(dy), mean(dy y)) are xor-shuffle reductions over the full warp, executed by all 32 lanes on warp-uniform control flow (a row
// past R is computed on zeros and not stored). No shared memory, no block barrier, no atomics, no communication between warps or
// blocks; every output element is written exactly once, by one lane. The results are a fixed function of the inputs (a fixed
// reduction tree): bitwise repeatable for any grid, allocator state or concurrent work.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>

#include <string>
#include <tuple>
#include <vector>

#include "../../conditioned_transition/cuda/token_dit_common.cuh"

namespace {
using namespace tdr;
using bf = __nv_bfloat16;

constexpr int WPB = 8;          // warps per block
constexpr int RPW = 2;          // rows per warp

// Y's row: y itself (fp32, CW columns) or [bf16(y) | bf16(y - bf16(y))] (bf16, 2 CW columns); CW the pair width (128 or 256: CW / 128
// float4 per lane, chunk c at columns 128 c + 4 lane)
template <typename YT, int CW> struct YOp;
template <int CW> struct YOp<float, CW> {
  static constexpr int COLS = CW;
  static __device__ __forceinline__ void store(float* row, int col, float4 y) { V4<float>::store(row + col, y); }
};
template <int CW> struct YOp<bf, CW> {
  static constexpr int COLS = 2 * CW;
  static __device__ __forceinline__ float rnd(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }
  static __device__ __forceinline__ void store(bf* row, int col, float4 y) {
    const float4 hi = make_float4(rnd(y.x), rnd(y.y), rnd(y.z), rnd(y.w));     // exact in bf16: the store below keeps it
    V4<bf>::store(row + col, hi);
    V4<bf>::store(row + CW + col, make_float4(y.x - hi.x, y.y - hi.y, y.z - hi.z, y.w - hi.w));   // y - hi exact in fp32
  }
};

template <int CW, typename XT, typename OT, typename YT>
__global__ void __launch_bounds__(WPB * 32) ln0_bwd_rows_kernel(const XT* __restrict__ X, const float* __restrict__ DY,
    OT* __restrict__ DX, YT* __restrict__ Y, long R, float eps) {
  static_assert(CW == 128 || CW == 256, "pair width 128 or 256");
  constexpr int K = CW / 128;
  const long row0 = ((long)blockIdx.x * WPB + threadIdx.x / 32) * RPW;
  const int lane = threadIdx.x % 32;
  float4 x[RPW][K], d[RPW][K];
#pragma unroll
  for (int k = 0; k < RPW; ++k) {
    const long r = row0 + k;
#pragma unroll
    for (int c = 0; c < K; ++c) {
      if (r < R) {
        x[k][c] = V4<XT>::load(X + r * CW + c * 128 + lane * 4);
        d[k][c] = V4<float>::load(DY + r * CW + c * 128 + lane * 4);
      } else {
        x[k][c] = make_float4(0.f, 0.f, 0.f, 0.f);
        d[k][c] = make_float4(0.f, 0.f, 0.f, 0.f);
      }
    }
  }
#pragma unroll
  for (int k = 0; k < RPW; ++k) {
    // the forward's statistics (layernorm_rows_kernel<CW>: the same sums in the same order), then y
    float4 y[K];
    float s = 0.f;
#pragma unroll
    for (int c = 0; c < K; ++c) { y[c] = x[k][c]; s += y[c].x + y[c].y + y[c].z + y[c].w; }
    const float mean = warp_sum(s) / (float)CW;
    float q = 0.f;
#pragma unroll
    for (int c = 0; c < K; ++c) {
      y[c].x -= mean; y[c].y -= mean; y[c].z -= mean; y[c].w -= mean;
      q += y[c].x * y[c].x + y[c].y * y[c].y + y[c].z * y[c].z + y[c].w * y[c].w;
    }
    const float rstd = rsqrtf(warp_sum(q) / (float)CW + eps);
    float s1 = 0.f, s2 = 0.f;
#pragma unroll
    for (int c = 0; c < K; ++c) {
      y[c].x *= rstd; y[c].y *= rstd; y[c].z *= rstd; y[c].w *= rstd;
      const float4 g = d[k][c];
      s1 += g.x + g.y + g.z + g.w;
      s2 += g.x * y[c].x + g.y * y[c].y + g.z * y[c].z + g.w * y[c].w;
    }
    const float m1 = warp_sum(s1) / (float)CW;                                       // mean(dy)
    const float m2 = warp_sum(s2) / (float)CW;                                       // mean(dy y)
    const long r = row0 + k;
#pragma unroll
    for (int c = 0; c < K; ++c) {
      const float4 g = d[k][c];
      const float4 o = make_float4(rstd * (g.x - m1 - y[c].x * m2), rstd * (g.y - m1 - y[c].y * m2), rstd * (g.z - m1 - y[c].z * m2),
                                   rstd * (g.w - m1 - y[c].w * m2));
      if (r < R) {
        V4<OT>::store(DX + r * CW + c * 128 + lane * 4, o);
        YOp<YT, CW>::store(Y + r * YOp<YT, CW>::COLS, c * 128 + lane * 4, y[c]);
      }
    }
  }
}

// y = LN0(x) in fp32 (no affine) for the EXACT pair-bias logits (user rule 2026-11-19: the bias-only DiT's logits and their softmax are fp32;
// bf16 / TF32 / split-product operands lost peptide bonds). x bf16 or fp32 [R, C], y fp32 [R, C]; ``PST`` (float2 [R]: mean, rstd; may be
// null) is what the pair_bias backward kernels read. Same sums in the same order as ln0_bwd_rows_kernel, so the backward's y is this
// kernel's. One warp per RPW rows, no shared memory, no barrier.
template <int CW, typename XT>
__global__ void __launch_bounds__(WPB * 32) ln0_fwd_rows_kernel(const XT* __restrict__ X, float* __restrict__ Y, float2* __restrict__ PST,
    long R, float eps) {
  constexpr int K = CW / 128;
  const long row0 = ((long)blockIdx.x * WPB + threadIdx.x / 32) * RPW;
  const int lane = threadIdx.x % 32;
  float4 x[RPW][K];
#pragma unroll
  for (int k = 0; k < RPW; ++k)
#pragma unroll
    for (int c = 0; c < K; ++c)
      x[k][c] = row0 + k < R ? V4<XT>::load(X + (row0 + k) * CW + c * 128 + lane * 4) : make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
  for (int k = 0; k < RPW; ++k) {
    float4 y[K];
    float s = 0.f;
#pragma unroll
    for (int c = 0; c < K; ++c) { y[c] = x[k][c]; s += y[c].x + y[c].y + y[c].z + y[c].w; }
    const float mean = warp_sum(s) / (float)CW;
    float q = 0.f;
#pragma unroll
    for (int c = 0; c < K; ++c) {
      y[c].x -= mean; y[c].y -= mean; y[c].z -= mean; y[c].w -= mean;
      q += y[c].x * y[c].x + y[c].y * y[c].y + y[c].z * y[c].z + y[c].w * y[c].w;
    }
    const float rstd = rsqrtf(warp_sum(q) / (float)CW + eps);
    const long r = row0 + k;
    if (r < R) {
#pragma unroll
      for (int c = 0; c < K; ++c)
        V4<float>::store(Y + r * CW + c * 128 + lane * 4, make_float4(y[c].x * rstd, y[c].y * rstd, y[c].z * rstd, y[c].w * rstd));
      if (PST != nullptr && lane == 0) PST[r] = make_float2(mean, rstd);
    }
  }
}

// The exact fp32 hoist backward of ONE block (N = 16 heads), every product exact fp32 on the FMA pipe (user rule 2026-11-19), in two passes over the
// pair rows that never write d LN0 (R x 128 fp32) or Y: the row statistics (mean, rstd: ln0_fwd's, saved by the forward) come in as PST [R] float2.
//   dln_bwd_kernel   d LN0 = dbias^T W' (16 FMAs a channel) and the LN0 backward -> d pair
//   dwf_kernel       dW' = dbias Y, Y = (x - mean) rstd rebuilt per row: a partial [16, 128] per warp over its rows in a fixed order; the caller sums
//                    the warps' partials in a fixed order (torch sum)
// A warp deals itself tiles of 32 rows round robin (nw warps, ~2 blocks of 8 an SM), a lane owns channels 4 lane .. 4 lane + 3; dbias [16, R] fp32 is read 32 rows at a time (coalesced) into a
// per-warp shared tile, W' sits in shared memory. No atomics, no communication between warps: bitwise repeatable for any grid.
template <int NN, int RG, typename XT, typename OT>
__global__ void __launch_bounds__(WPB * 32, 2) dln_bwd_kernel(const XT* __restrict__ X, const float2* __restrict__ PST, const float* __restrict__ DB, long dbs,
    const float* __restrict__ WF, OT* __restrict__ DX, long R) {
  __shared__ float4 sw[NN * 32];
  __shared__ float4 sdb[WPB][32][NN / 4];
  const int lane = threadIdx.x % 32, wi = threadIdx.x / 32;
  for (int t = threadIdx.x; t < NN * 32; t += WPB * 32) sw[t] = reinterpret_cast<const float4*>(WF)[t];
  __syncthreads();
  const long NWt = (long)gridDim.x * WPB;
  for (long tile = (long)blockIdx.x * WPB + wi; tile * 32 < R; tile += NWt) {   // R % 32 == 0 (checked by the caller): whole tiles of 32 rows, tiles dealt round robin
    const long r0 = tile * 32;
    {
      float v[NN];
#pragma unroll
      for (int n = 0; n < NN; ++n) v[n] = DB[(long)n * dbs + r0 + lane];
#pragma unroll
      for (int q = 0; q < NN / 4; ++q) sdb[wi][lane][q] = make_float4(v[4 * q], v[4 * q + 1], v[4 * q + 2], v[4 * q + 3]);
    }
    __syncwarp();
    float4 xn[RG];
    float2 pn[RG];
#pragma unroll
    for (int k = 0; k < RG; ++k) { xn[k] = V4<XT>::load(X + (r0 + k) * 128 + lane * 4); pn[k] = PST[r0 + k]; }
    for (int i = 0; i < 32; i += RG) {
      float4 y[RG], g[RG];
      float rstd[RG], s1[RG], s2[RG];
#pragma unroll
      for (int k = 0; k < RG; ++k) {
        y[k] = xn[k]; rstd[k] = pn[k].y;
        const float mean = pn[k].x;
        if (i + RG < 32) { xn[k] = V4<XT>::load(X + (r0 + i + RG + k) * 128 + lane * 4); pn[k] = PST[r0 + i + RG + k]; }
        y[k].x -= mean; y[k].y -= mean; y[k].z -= mean; y[k].w -= mean;
        y[k].x *= rstd[k]; y[k].y *= rstd[k]; y[k].z *= rstd[k]; y[k].w *= rstd[k];
        g[k] = make_float4(0.f, 0.f, 0.f, 0.f);                // d LN0 = sum_n dbias[n] W'[n]
#pragma unroll
        for (int q = 0; q < NN / 4; ++q) {
          const float4 dq = sdb[wi][i + k][q];
          const float dv[4] = {dq.x, dq.y, dq.z, dq.w};
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const float4 wv = sw[(4 * q + e) * 32 + lane];
            g[k].x = fmaf(dv[e], wv.x, g[k].x); g[k].y = fmaf(dv[e], wv.y, g[k].y); g[k].z = fmaf(dv[e], wv.z, g[k].z); g[k].w = fmaf(dv[e], wv.w, g[k].w);
          }
        }
        s1[k] = g[k].x + g[k].y + g[k].z + g[k].w;
        s2[k] = g[k].x * y[k].x + g[k].y * y[k].y + g[k].z * y[k].z + g[k].w * y[k].w;
      }
#pragma unroll
      for (int k = 0; k < RG; ++k) {
        float a = s1[k], b = s2[k];
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) { a += __shfl_xor_sync(0xffffffffu, a, o); b += __shfl_xor_sync(0xffffffffu, b, o); }
        a /= 128.f; b /= 128.f;                                // mean(dy), mean(dy y): warp_sum's xor tree
        V4<OT>::store(DX + (r0 + i + k) * 128 + lane * 4,
                      make_float4(rstd[k] * (g[k].x - a - y[k].x * b), rstd[k] * (g[k].y - a - y[k].y * b),
                                  rstd[k] * (g[k].z - a - y[k].z * b), rstd[k] * (g[k].w - a - y[k].w * b)));
      }
    }
    __syncwarp();
  }
}

template <int NN, int RG, typename XT>
__global__ void __launch_bounds__(WPB * 32, 2) dwf_kernel(const XT* __restrict__ X, const float2* __restrict__ PST, const float* __restrict__ DB, long dbs,
    float* __restrict__ PART, long R) {
  __shared__ float4 sdb[WPB][32][NN / 4];
  const int lane = threadIdx.x % 32, wi = threadIdx.x / 32;
  const long w = (long)blockIdx.x * WPB + wi;
  const long NWt = (long)gridDim.x * WPB;
  float acc[NN][4];
#pragma unroll
  for (int n = 0; n < NN; ++n) { acc[n][0] = acc[n][1] = acc[n][2] = acc[n][3] = 0.f; }
  for (long tile = w; tile * 32 < R; tile += NWt) {
    const long r0 = tile * 32;
    {
      float v[NN];
#pragma unroll
      for (int n = 0; n < NN; ++n) v[n] = DB[(long)n * dbs + r0 + lane];
#pragma unroll
      for (int q = 0; q < NN / 4; ++q) sdb[wi][lane][q] = make_float4(v[4 * q], v[4 * q + 1], v[4 * q + 2], v[4 * q + 3]);
    }
    __syncwarp();
    float4 xn[RG];
    float2 pn[RG];
#pragma unroll
    for (int k = 0; k < RG; ++k) { xn[k] = V4<XT>::load(X + (r0 + k) * 128 + lane * 4); pn[k] = PST[r0 + k]; }
    for (int i = 0; i < 32; i += RG) {
      float4 y[RG];
#pragma unroll
      for (int k = 0; k < RG; ++k) {
        y[k] = xn[k];
        const float mean = pn[k].x, rstd = pn[k].y;
        if (i + RG < 32) { xn[k] = V4<XT>::load(X + (r0 + i + RG + k) * 128 + lane * 4); pn[k] = PST[r0 + i + RG + k]; }
        y[k].x -= mean; y[k].y -= mean; y[k].z -= mean; y[k].w -= mean;
        y[k].x *= rstd; y[k].y *= rstd; y[k].z *= rstd; y[k].w *= rstd;
      }
#pragma unroll
      for (int k = 0; k < RG; ++k)
#pragma unroll
        for (int q = 0; q < NN / 4; ++q) {
          const float4 dq = sdb[wi][i + k][q];
          const float dv[4] = {dq.x, dq.y, dq.z, dq.w};
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            acc[4 * q + e][0] = fmaf(dv[e], y[k].x, acc[4 * q + e][0]); acc[4 * q + e][1] = fmaf(dv[e], y[k].y, acc[4 * q + e][1]);
            acc[4 * q + e][2] = fmaf(dv[e], y[k].z, acc[4 * q + e][2]); acc[4 * q + e][3] = fmaf(dv[e], y[k].w, acc[4 * q + e][3]);
          }
        }
    }
    __syncwarp();
  }
#pragma unroll
  for (int n = 0; n < NN; ++n)
    reinterpret_cast<float4*>(PART + (w * NN + n) * 128)[lane] = make_float4(acc[n][0], acc[n][1], acc[n][2], acc[n][3]);
}

// The exact fp32 pair-bias LOGITS of one block (N = 16 heads) in one pass over the pair rows: logits[n, r] = sum_c W'[n, c] LN0(x[r])[c], every product
// and sum fp32 on the FMA pipe (user rule 2026-11-19), without LN0 (R x 128 fp32) ever being stored. A lane owns ONE ROW (a warp: 32 rows staged
// through shared memory, rows 132 floats apart: conflict-free float4 reads); the row statistics follow ln0_fwd_rows_kernel's arithmetic (the lane
// partials of 4 channels, then warp_sum's xor-butterfly, written out as the same tree over 32 partials); the dot products run c = 0 .. 127 in
// order. PST (float2 [R]: mean, rstd; may be null).
template <typename XT>
__global__ void __launch_bounds__(64) logits_fused_kernel(const XT* __restrict__ X, const float* __restrict__ WF, float* __restrict__ LG,
    float2* __restrict__ PST, long R, float eps) {
  constexpr int NN = 16, SR = 132;
  __shared__ float4 sw[NN * 32];
  __shared__ __align__(16) float sx[2][32 * SR];
  const int lane = threadIdx.x % 32, wi = threadIdx.x / 32;
  for (int t = threadIdx.x; t < NN * 32; t += 64) sw[t] = reinterpret_cast<const float4*>(WF)[t];
  __syncthreads();
  constexpr int TPW = 4;
  const long w = (long)blockIdx.x * 2 + wi;
  for (int tile = 0; tile < TPW; ++tile) {
    const long r0 = (w * TPW + tile) * 32;
    if (r0 >= R) return;
    for (int i = 0; i < 32; ++i)
      *reinterpret_cast<float4*>(&sx[wi][i * SR + lane * 4]) = V4<XT>::load(X + (r0 + i) * 128 + lane * 4);
    __syncwarp();
    const float* xr = &sx[wi][lane * SR];
    float p[32];
#pragma unroll
    for (int j = 0; j < 32; ++j) { const float4 v = *reinterpret_cast<const float4*>(xr + 4 * j); p[j] = 0.f + v.x + v.y + v.z + v.w; }
#pragma unroll
    for (int i = 0; i < 16; ++i) p[i] = p[i] + p[i + 16];
#pragma unroll
    for (int i = 0; i < 8; ++i) p[i] = p[i] + p[i + 8];
#pragma unroll
    for (int i = 0; i < 4; ++i) p[i] = p[i] + p[i + 4];
#pragma unroll
    for (int i = 0; i < 2; ++i) p[i] = p[i] + p[i + 2];
    p[0] = p[0] + p[1];
    const float mean = p[0] / 128.f;
#pragma unroll
    for (int j = 0; j < 32; ++j) {
      float4 v = *reinterpret_cast<const float4*>(xr + 4 * j);
      v.x -= mean; v.y -= mean; v.z -= mean; v.w -= mean;
      p[j] = 0.f + (v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w);
    }
#pragma unroll
    for (int i = 0; i < 16; ++i) p[i] = p[i] + p[i + 16];
#pragma unroll
    for (int i = 0; i < 8; ++i) p[i] = p[i] + p[i + 8];
#pragma unroll
    for (int i = 0; i < 4; ++i) p[i] = p[i] + p[i + 4];
#pragma unroll
    for (int i = 0; i < 2; ++i) p[i] = p[i] + p[i + 2];
    p[0] = p[0] + p[1];
    const float rstd = rsqrtf(p[0] / 128.f + eps);
    float acc[NN];
#pragma unroll
    for (int n = 0; n < NN; ++n) acc[n] = 0.f;
#pragma unroll 4
    for (int j = 0; j < 32; ++j) {
      float4 y = *reinterpret_cast<const float4*>(xr + 4 * j);
      y.x -= mean; y.y -= mean; y.z -= mean; y.w -= mean;
      y.x *= rstd; y.y *= rstd; y.z *= rstd; y.w *= rstd;
#pragma unroll
      for (int n = 0; n < NN; ++n) {
        const float4 wv = sw[n * 32 + j];
        acc[n] = fmaf(wv.x, y.x, acc[n]); acc[n] = fmaf(wv.y, y.y, acc[n]); acc[n] = fmaf(wv.z, y.z, acc[n]); acc[n] = fmaf(wv.w, y.w, acc[n]);
      }
    }
    const long r = r0 + lane;
#pragma unroll
    for (int n = 0; n < NN; ++n) LG[(long)n * R + r] = acc[n];
    if (PST != nullptr) PST[r] = make_float2(mean, rstd);
    __syncwarp();
  }
}


// The softmax-gradient epilogue of the bf16 bias-only attention's EXACT fp32 bias gradient (user rule 2026-11-19), one pass: dbias = P o (dP - D) with
// P = softmax(logits) re-formed in fp32 from the saved fp32 logits (key mask folded in: masked keys P = 0, as torch.softmax of the filled logits) and
// D = sum_j P dP the exact fp32 row sum. dP [H L, L] fp32 is the batched GEMM's output (OUT may alias it). One warp per row; the row (L = 128 KC keys,
// lane: 4 consecutive keys of every 128) lives in registers: logits and dP are read once, the result written once. Fixed order everywhere (lane
// partials in key order, then warp_sum's xor tree): bitwise repeatable.
template <int KC>
__global__ void __launch_bounds__(128) dbias_rows_kernel(const float* __restrict__ LG, const float* __restrict__ DP, float* __restrict__ OUT, const bool* __restrict__ MASK,
    long rows, int L, bool pin) {
  const int lane = threadIdx.x % 32;
  const long row = (long)blockIdx.x * 4 + threadIdx.x / 32;
  if (row >= rows) return;
  const float NEG = -3.4028234663852886e38f;
  float4 l[KC], d[KC];
#pragma unroll
  for (int k = 0; k < KC; ++k) {
    l[k] = *reinterpret_cast<const float4*>(LG + row * L + k * 128 + lane * 4);
    d[k] = *reinterpret_cast<const float4*>(DP + row * L + k * 128 + lane * 4);
  }
  if (!pin) {   // LG are logits: P = softmax(logits with the masked keys at -FLT_MAX); pin: LG already is P (fp32, masked keys 0)
    float m = NEG;
#pragma unroll
    for (int k = 0; k < KC; ++k) {
      if (MASK) {
        const int j = k * 128 + lane * 4;
        l[k].x = MASK[j] ? l[k].x : NEG; l[k].y = MASK[j + 1] ? l[k].y : NEG; l[k].z = MASK[j + 2] ? l[k].z : NEG; l[k].w = MASK[j + 3] ? l[k].w : NEG;
      }
      m = fmaxf(m, fmaxf(fmaxf(l[k].x, l[k].y), fmaxf(l[k].z, l[k].w)));
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    float z = 0.f;
#pragma unroll
    for (int k = 0; k < KC; ++k) {
      l[k].x = expf(l[k].x - m); l[k].y = expf(l[k].y - m); l[k].z = expf(l[k].z - m); l[k].w = expf(l[k].w - m);   // e_j
      z += l[k].x + l[k].y + l[k].z + l[k].w;
    }
    z = warp_sum(z);
#pragma unroll
    for (int k = 0; k < KC; ++k) { l[k].x /= z; l[k].y /= z; l[k].z /= z; l[k].w /= z; }                          // P
  }
  float s = 0.f;
#pragma unroll
  for (int k = 0; k < KC; ++k) s += l[k].x * d[k].x + l[k].y * d[k].y + l[k].z * d[k].z + l[k].w * d[k].w;
  s = warp_sum(s);                                                                                                   // D = sum_j P dP
#pragma unroll
  for (int k = 0; k < KC; ++k)
    *reinterpret_cast<float4*>(OUT + row * L + k * 128 + lane * 4) =
        make_float4(l[k].x * (d[k].x - s), l[k].y * (d[k].y - s), l[k].z * (d[k].z - s), l[k].w * (d[k].w - s));
}

// do, v [A L, H dh] bf16 (sample-major rows, head-major columns; any row stride) -> head-major [H, L, A dh] copies of both (the operands of the bias gradient's
// batched GEMM dP[h] = dO[h] V[h]^T, K = (a, d)): one block per source row, 16-byte chunks; a row of a head's slab is written as A runs of dh contiguous elements
__global__ void __launch_bounds__(128) permute_dv_kernel(const int4* __restrict__ DO, long ldo, const int4* __restrict__ V, long ldv, int4* __restrict__ ODO,
    int4* __restrict__ OV, int A, int H, int L, int dh8) {
  const int t = threadIdx.x;
  if (t >= H * dh8) return;
  const long r = blockIdx.x;                       // a L + l
  const int a = (int)(r / L), l = (int)(r % L), h = t / dh8, j = t % dh8;
  const long dst = (((long)h * L + l) * A + a) * dh8 + j;
  ODO[dst] = DO[r * ldo + t];
  OV[dst] = V[r * ldv + t];
}

void permute_dv(at::Tensor dO, at::Tensor v, at::Tensor odo, at::Tensor ov, int64_t A, int64_t H, int64_t L) {
  const int64_t DAc = dO.size(1), dh = DAc / H;
  TORCH_CHECK(dO.is_cuda() && v.is_cuda() && dO.scalar_type() == at::kBFloat16 && v.scalar_type() == at::kBFloat16 && dO.dim() == 2 && v.dim() == 2 && dO.size(0) == A * L &&
              v.size(0) == A * L && v.size(1) == DAc && dO.stride(1) == 1 && v.stride(1) == 1 && DAc % H == 0 && dh % 8 == 0 && H * (dh / 8) <= 128, "permute_dv: do, v bf16 [A L, H dh], dh a multiple of 8");
  TORCH_CHECK(dO.stride(0) % 8 == 0 && v.stride(0) % 8 == 0 && reinterpret_cast<uintptr_t>(dO.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(v.data_ptr()) % 16 == 0,
              "permute_dv: 16-byte aligned rows");
  TORCH_CHECK(odo.is_cuda() && ov.is_cuda() && odo.is_contiguous() && ov.is_contiguous() && odo.scalar_type() == at::kBFloat16 && ov.scalar_type() == at::kBFloat16 &&
              odo.numel() == A * L * DAc && ov.numel() == A * L * DAc, "permute_dv: outputs bf16 [H, L, A dh]");
  const at::cuda::CUDAGuard g(dO.device());
  permute_dv_kernel<<<(unsigned)(A * L), 128, 0, at::cuda::getCurrentCUDAStream()>>>(reinterpret_cast<const int4*>(dO.data_ptr()), (long)(dO.stride(0) / 8),
      reinterpret_cast<const int4*>(v.data_ptr()), (long)(v.stride(0) / 8), reinterpret_cast<int4*>(odo.data_ptr()), reinterpret_cast<int4*>(ov.data_ptr()), (int)A, (int)H, (int)L, (int)(dh / 8));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

#define HR_DISPATCH(T, NAME, ...)                                                      \
  [&] {                                                                                              \
    if ((T) == at::kFloat) { using NAME = float; return __VA_ARGS__(); }                             \
    TORCH_CHECK((T) == at::kBFloat16, "bf16 or fp32");                                                 \
    using NAME = bf; return __VA_ARGS__();                                                           \
  }()

void rows(const at::Tensor& t, const char* name, int64_t R, int64_t cols) {
  TORCH_CHECK(t.is_cuda() && t.dim() == 2 && t.size(0) == R && t.size(1) == cols && t.is_contiguous(), name, ": contiguous [R, ",
              cols, "]");
  TORCH_CHECK(t.scalar_type() == at::kFloat || t.scalar_type() == at::kBFloat16, name, ": bf16 or fp32");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0, name, ": 16-byte aligned");
}

// dx = LN0 backward of dy at x (no affine) and Y = the dW' GEMM operand of y = LN0(x): x, dx bf16 or fp32 [R, C], dy fp32 [R, C],
// y bf16 [R, 2 C] (hi | lo) or fp32 [R, C]; all contiguous; C the pair width, 128 or 256
void ln0_bwd_rows(at::Tensor x, at::Tensor dy, at::Tensor dx, at::Tensor y, double eps) {
  const int64_t R = x.size(0), CW = x.dim() == 2 ? x.size(1) : 0;
  TORCH_CHECK(CW == 128 || CW == 256, "ln0_bwd_rows: rows of 128 or 256 channels");
  rows(x, "x", R, CW); rows(dy, "dy", R, CW); rows(dx, "dx", R, CW);
  rows(y, "y", R, y.scalar_type() == at::kBFloat16 ? 2 * CW : CW);
  TORCH_CHECK(dy.scalar_type() == at::kFloat, "ln0_bwd_rows: dy fp32");
  TORCH_CHECK(x.device() == dy.device() && x.device() == dx.device() && x.device() == y.device(), "ln0_bwd_rows: one device");
  if (R == 0) return;
  const at::cuda::CUDAGuard g(x.device());
  const int64_t per = (int64_t)WPB * RPW;
  HR_DISPATCH(x.scalar_type(), XT, [&] {
    HR_DISPATCH(dx.scalar_type(), OT, [&] {
      HR_DISPATCH(y.scalar_type(), YT, [&] {
        auto k = CW == 128 ? ln0_bwd_rows_kernel<128, XT, OT, YT> : ln0_bwd_rows_kernel<256, XT, OT, YT>;
        k<<<(unsigned)((R + per - 1) / per), WPB * 32, 0, at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<const XT*>(x.data_ptr()), reinterpret_cast<const float*>(dy.data_ptr()),
            reinterpret_cast<OT*>(dx.data_ptr()), reinterpret_cast<YT*>(y.data_ptr()), (long)R, (float)eps);
      });
    });
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// y = LN0(x) fp32 [R, C] from x bf16 / fp32 [R, C] (C = 128 or 256); pst float [R, 2] (mean, rstd) or an empty tensor
void ln0_fwd_rows(at::Tensor x, at::Tensor y, at::Tensor pst, double eps) {
  const int64_t R = x.size(0), CW = x.dim() == 2 ? x.size(1) : 0;
  TORCH_CHECK(CW == 128 || CW == 256, "ln0_fwd_rows: rows of 128 or 256 channels");
  rows(x, "x", R, CW); rows(y, "y", R, CW);
  TORCH_CHECK(y.scalar_type() == at::kFloat, "ln0_fwd_rows: y fp32");
  const bool stats = pst.numel() != 0;
  TORCH_CHECK(!stats || (pst.is_cuda() && pst.is_contiguous() && pst.scalar_type() == at::kFloat && pst.numel() == 2 * R),
              "ln0_fwd_rows: pst float [R, 2]");
  if (R == 0) return;
  const at::cuda::CUDAGuard g(x.device());
  const int64_t per = (int64_t)WPB * RPW;
  HR_DISPATCH(x.scalar_type(), XT, [&] {
    auto k = CW == 128 ? ln0_fwd_rows_kernel<128, XT> : ln0_fwd_rows_kernel<256, XT>;
    k<<<(unsigned)((R + per - 1) / per), WPB * 32, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const XT*>(x.data_ptr()), reinterpret_cast<float*>(y.data_ptr()),
        stats ? reinterpret_cast<float2*>(pst.data_ptr()) : nullptr, (long)R, (float)eps);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// dx [R, 128] and part [nw, 16, 128] (nw warps, a multiple of 8: tiles of 32 rows dealt round robin) from x [R, 128] (bf16 / fp32), the saved row statistics pst float [R, 2] (mean, rstd), dbias [16, R] fp32
// (row stride any, columns contiguous) and W' [16, 128] fp32; R a multiple of 32
void hoist_bwd_fused(at::Tensor x, at::Tensor db, at::Tensor wf, at::Tensor pst, at::Tensor dx, at::Tensor part, int64_t nw, int64_t rg) {
  const int64_t R = x.size(0);
  TORCH_CHECK(x.dim() == 2 && x.size(1) == 128 && R % 32 == 0, "hoist_bwd_fused: rows of 128 channels, R a multiple of 32");
  rows(x, "x", R, 128); rows(dx, "dx", R, 128);
  TORCH_CHECK(db.is_cuda() && db.dim() == 2 && db.size(0) == 16 && db.size(1) == R && db.stride(1) == 1 && db.scalar_type() == at::kFloat,
              "hoist_bwd_fused: dbias fp32 [16, R], contiguous columns");
  TORCH_CHECK(wf.is_cuda() && wf.is_contiguous() && wf.scalar_type() == at::kFloat && wf.dim() == 2 && wf.size(0) == 16 && wf.size(1) == 128,
              "hoist_bwd_fused: W' fp32 [16, 128]");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(wf.data_ptr()) % 16 == 0, "hoist_bwd_fused: W' 16-byte aligned");
  TORCH_CHECK(pst.is_cuda() && pst.is_contiguous() && pst.scalar_type() == at::kFloat && pst.numel() == 2 * R, "hoist_bwd_fused: pst float [R, 2]");
  TORCH_CHECK(nw > 0 && nw % WPB == 0, "hoist_bwd_fused: nw a multiple of 8");
  TORCH_CHECK(part.is_cuda() && part.is_contiguous() && part.scalar_type() == at::kFloat && part.numel() == nw * 16 * 128, "hoist_bwd_fused: part [nw, 16, 128]");
  if (R == 0) return;
  const at::cuda::CUDAGuard g(x.device());
  const unsigned grid = (unsigned)((nw + WPB - 1) / WPB);
  auto st = at::cuda::getCurrentCUDAStream();
  const float2* pp = reinterpret_cast<const float2*>(pst.data_ptr());
  HR_DISPATCH(x.scalar_type(), XT, [&] {
    HR_DISPATCH(dx.scalar_type(), OT, [&] {
      const XT* xp = reinterpret_cast<const XT*>(x.data_ptr());
      OT* dp = reinterpret_cast<OT*>(dx.data_ptr());
      if (rg == 2) dln_bwd_kernel<16, 2, XT, OT><<<grid, WPB * 32, 0, st>>>(xp, pp, db.data_ptr<float>(), (long)db.stride(0), wf.data_ptr<float>(), dp, (long)R);
      else if (rg == 1) dln_bwd_kernel<16, 1, XT, OT><<<grid, WPB * 32, 0, st>>>(xp, pp, db.data_ptr<float>(), (long)db.stride(0), wf.data_ptr<float>(), dp, (long)R);
      else dln_bwd_kernel<16, 4, XT, OT><<<grid, WPB * 32, 0, st>>>(xp, pp, db.data_ptr<float>(), (long)db.stride(0), wf.data_ptr<float>(), dp, (long)R);
    });
    if (rg == 2) dwf_kernel<16, 2, XT><<<grid, WPB * 32, 0, st>>>(reinterpret_cast<const XT*>(x.data_ptr()), pp, db.data_ptr<float>(), (long)db.stride(0), part.data_ptr<float>(), (long)R);
    else if (rg == 1) dwf_kernel<16, 1, XT><<<grid, WPB * 32, 0, st>>>(reinterpret_cast<const XT*>(x.data_ptr()), pp, db.data_ptr<float>(), (long)db.stride(0), part.data_ptr<float>(), (long)R);
    else dwf_kernel<16, 4, XT><<<grid, WPB * 32, 0, st>>>(reinterpret_cast<const XT*>(x.data_ptr()), pp, db.data_ptr<float>(), (long)db.stride(0), part.data_ptr<float>(), (long)R);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// out [H L, L] fp32 <- P o (dp - D) (out may be dp) from the fp32 logits lg [H L, L] and the key mask (bool [L] or an empty tensor); L = 128 KC, KC 1 .. 8
void dbias_rows(at::Tensor lg, at::Tensor dp, at::Tensor out, at::Tensor mask, bool pin) {
  const int64_t L = lg.dim() == 2 ? lg.size(1) : 0, rows_ = lg.size(0);
  TORCH_CHECK(L % 128 == 0 && L >= 128 && L <= 1024 && lg.size(0) % L == 0, "dbias_rows: L a multiple of 128, at most 1024");
  TORCH_CHECK(lg.is_cuda() && dp.is_cuda() && lg.is_contiguous() && dp.is_contiguous() && lg.scalar_type() == at::kFloat && dp.scalar_type() == at::kFloat &&
              dp.sizes() == lg.sizes() && out.is_cuda() && out.is_contiguous() && out.scalar_type() == at::kFloat && out.sizes() == lg.sizes() && reinterpret_cast<uintptr_t>(out.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(lg.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(dp.data_ptr()) % 16 == 0,
              "dbias_rows: logits, dP fp32 [H L, L]");
  const bool has_mask = mask.numel() != 0;
  TORCH_CHECK(!has_mask || (mask.is_cuda() && mask.is_contiguous() && mask.scalar_type() == at::kBool && mask.numel() == L), "dbias_rows: mask bool [L]");
  const at::cuda::CUDAGuard g(lg.device());
  const unsigned blocks = (unsigned)((rows_ + 3) / 4);
  auto st = at::cuda::getCurrentCUDAStream();
  const bool* mp = has_mask ? mask.data_ptr<bool>() : nullptr;
#define DBR(K) case K: dbias_rows_kernel<K><<<blocks, 128, 0, st>>>(lg.data_ptr<float>(), dp.data_ptr<float>(), out.data_ptr<float>(), mp, rows_, (int)L, pin); break
  switch (L / 128) { DBR(1); DBR(2); DBR(3); DBR(4); DBR(5); DBR(6); DBR(7); DBR(8); default: TORCH_CHECK(false, "dbias_rows: L"); }
#undef DBR
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// lg [16, R] fp32 = W' LN0(x)^T from x [R, 128] (bf16 / fp32), W' [16, 128] fp32; pst float [R, 2] (mean, rstd) or an empty tensor; R a multiple of 32
void logits_fused(at::Tensor x, at::Tensor wf, at::Tensor lg, at::Tensor pst, double eps) {
  const int64_t R = x.size(0);
  TORCH_CHECK(x.dim() == 2 && x.size(1) == 128 && R % 32 == 0, "logits_fused: rows of 128 channels, R a multiple of 32");
  rows(x, "x", R, 128);
  TORCH_CHECK(wf.is_cuda() && wf.is_contiguous() && wf.scalar_type() == at::kFloat && wf.dim() == 2 && wf.size(0) == 16 && wf.size(1) == 128 &&
              reinterpret_cast<uintptr_t>(wf.data_ptr()) % 16 == 0, "logits_fused: W' fp32 [16, 128]");
  TORCH_CHECK(lg.is_cuda() && lg.is_contiguous() && lg.scalar_type() == at::kFloat && lg.numel() == 16 * R, "logits_fused: logits fp32 [16, R]");
  const bool stats = pst.numel() != 0;
  TORCH_CHECK(!stats || (pst.is_cuda() && pst.is_contiguous() && pst.scalar_type() == at::kFloat && pst.numel() == 2 * R), "logits_fused: pst float [R, 2]");
  if (R == 0) return;
  const at::cuda::CUDAGuard g(x.device());
  const int64_t warps = (R / 32 + 3) / 4;
  HR_DISPATCH(x.scalar_type(), XT, [&] {
    logits_fused_kernel<XT><<<(unsigned)((warps + 1) / 2), 64, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const XT*>(x.data_ptr()), wf.data_ptr<float>(), lg.data_ptr<float>(),
        stats ? reinterpret_cast<float2*>(pst.data_ptr()) : nullptr, (long)R, (float)eps);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// (name, registers, local-memory bytes) of every kernel of this extension (the tests hold them to no local memory)
std::vector<std::tuple<std::string, int64_t, int64_t>> func_attrs() {
  std::vector<std::tuple<std::string, int64_t, int64_t>> out;
  auto add = [&](const char* n, const void* f) {
    cudaFuncAttributes a{};
    TORCH_CHECK(cudaFuncGetAttributes(&a, f) == cudaSuccess, "cudaFuncGetAttributes ", n);
    out.emplace_back(n, (int64_t)a.numRegs, (int64_t)a.localSizeBytes);
  };
#define FA(CW, XT, OT, YT) add("ln0_bwd_rows_kernel<" #CW ", " #XT ", " #OT ", " #YT ">", (const void*)ln0_bwd_rows_kernel<CW, XT, OT, YT>)
  FA(128, bf, bf, bf); FA(128, bf, float, bf); FA(128, float, bf, bf); FA(128, float, float, bf);
  FA(128, bf, bf, float); FA(128, bf, float, float); FA(128, float, bf, float); FA(128, float, float, float);
  FA(256, bf, bf, bf); FA(256, bf, float, bf); FA(256, float, bf, bf); FA(256, float, float, bf);
  FA(256, bf, bf, float); FA(256, bf, float, float); FA(256, float, bf, float); FA(256, float, float, float);
#define FB(XT, OT) add("dln_bwd_kernel<16, 2, " #XT ", " #OT ">", (const void*)dln_bwd_kernel<16, 2, XT, OT>)
  FB(bf, bf); FB(bf, float); FB(float, bf); FB(float, float);
  add("dwf_kernel<16, 2, bf>", (const void*)dwf_kernel<16, 2, bf>);
  add("dwf_kernel<16, 2, float>", (const void*)dwf_kernel<16, 2, float>);
#undef FB
  add("permute_dv_kernel", (const void*)permute_dv_kernel);
  add("dbias_rows_kernel<3>", (const void*)dbias_rows_kernel<3>);
  add("dbias_rows_kernel<6>", (const void*)dbias_rows_kernel<6>);
  add("logits_fused_kernel<bf>", (const void*)logits_fused_kernel<bf>);
  add("logits_fused_kernel<float>", (const void*)logits_fused_kernel<float>);
  add("ln0_fwd_rows_kernel<128, bf>", (const void*)ln0_fwd_rows_kernel<128, bf>);
  add("ln0_fwd_rows_kernel<128, float>", (const void*)ln0_fwd_rows_kernel<128, float>);
  add("ln0_fwd_rows_kernel<256, bf>", (const void*)ln0_fwd_rows_kernel<256, bf>);
  add("ln0_fwd_rows_kernel<256, float>", (const void*)ln0_fwd_rows_kernel<256, float>);
#undef FA
  return out;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ln0_bwd_rows_cuda", &ln0_bwd_rows);
  m.def("ln0_fwd_rows_cuda", &ln0_fwd_rows);
  m.def("hoist_bwd_fused_cuda", &hoist_bwd_fused);
  m.def("logits_fused_cuda", &logits_fused);
  m.def("dbias_rows_cuda", &dbias_rows);
  m.def("permute_dv_cuda", &permute_dv);
  m.def("func_attrs", &func_attrs);
}
