// lslide.cu — AlphaFold 3's slid key window (``LocalDiTBlock(key_window="slide")``) on top of the clipped-window attention kernels.
//
// A slid window starts at S_w = min(max(32 w - 48, 0), max(n_real - 128, 0)) instead of 32 w - 48 (n_real: the real atoms, which come
// first; the trunked pair is laid out from the window's first key either way, so the per-window bias tensor is the same). Only the first
// two windows and the last three real windows can differ from the clipped window (a window ends at 32 w + 80), so the attention kernels
// run the clipped geometry and this kernel recomputes those five windows over their slid keys and overwrites their rows of O (and of the
// LSE, when given). The five window indices come from the real atom count on the device (the key mask's population count, or n0 without
// a mask): no host read, so the call is CUDA-graph capturable and a replay with another mask is right. A window listed twice is done once.
//
// One CTA per (listed window, head, sample), 512 threads. Scores: a thread per (key, query quarter) with its k row in registers, the q rows
// read as shared-memory broadcasts; softmax: a warp per 2 queries (max, exp, sum by shuffles); O = P V: a thread per (query, 2 channels).
// A window the slide does not move is skipped. Plain fp32 CUDA cores: five windows of
// 32 queries are noise next to the window kernels. Keys at or past n_real are masked (-inf); a query with no valid key outputs 0.
//   q, k, v: [A, N, *] rows ld elements apart (bf16: the projections, ld 128; fp32: column blocks of P32, ld 512), head h at 32 h;
//   bias [4, nwin, 32, 128] fp32; km [N] uint8 or null; o [A, N, 128] (q's dtype), lse [A, 4, N] fp32 (log2 units) or null.
// SPDX-License-Identifier: Apache-2.0
#include <cuda_bf16.h>
#include <cstdint>

#define DEVI __device__ __forceinline__
constexpr int NH = 4, DH = 32, DM = 128, WQ = 32, WK = 128, KOFF = 48;
constexpr float LOG2E = 1.4426950408889634f;
constexpr float QSCALE = 0.17677669529663687f;           // 1 / sqrt(32)
constexpr int NW = 16, QG = NW / 4;                      // warps per CTA, query groups of the score phase

DEVI void st_f(float* p, float v) { *p = v; }
DEVI void load16(const float* p, float (&x)[16]) {
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float4 t = __ldg(reinterpret_cast<const float4*>(p) + i);
    x[4 * i] = t.x; x[4 * i + 1] = t.y; x[4 * i + 2] = t.z; x[4 * i + 3] = t.w;
  }
}
DEVI void load16(const __nv_bfloat16* p, float (&x)[16]) {
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    const uint4 t = __ldg(reinterpret_cast<const uint4*>(p) + i);
    const uint32_t u[4] = {t.x, t.y, t.z, t.w};
#pragma unroll
    for (int j = 0; j < 4; ++j) { x[8 * i + 2 * j] = __uint_as_float(u[j] << 16); x[8 * i + 2 * j + 1] = __uint_as_float(u[j] & 0xffff0000u); }
  }
}
DEVI void st_f(__nv_bfloat16* p, float v) { *p = __float2bfloat16_rn(v); }

template <typename T>
DEVI void slide_edges(const T* __restrict__ q, const T* __restrict__ k, const T* __restrict__ v, int ld, const float* __restrict__ bias,
                      const uint8_t* __restrict__ km, T* __restrict__ o, float* __restrict__ lse, int N, int n0) {
  __shared__ __align__(16) float qs[WQ][DH], vs[WK][DH], ss[WQ][WK + 1];   // 45.1 KB (under the 48 KB static limit)
  __shared__ int red[NW];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int h = blockIdx.y, a = blockIdx.z, nwin = N / WQ;
  // ---- the real atom count
  int nr = n0;
  if (km != nullptr) {
    int c = 0;
    for (int i = 16 * tid; i < N; i += 16 * blockDim.x) {              // N is a multiple of 128: whole 16-byte groups
      const uint4 b = *reinterpret_cast<const uint4*>(km + i);
      const uint32_t w4[4] = {b.x, b.y, b.z, b.w};
#pragma unroll
      for (int j = 0; j < 4; ++j)
        c += ((w4[j] & 0xffu) != 0) + ((w4[j] & 0xff00u) != 0) + ((w4[j] & 0xff0000u) != 0) + ((w4[j] >> 24) != 0);
    }
#pragma unroll
    for (int off = 16; off >= 1; off >>= 1) c += __shfl_xor_sync(0xffffffffu, c, off);
    if (lane == 0) red[warp] = c;
    __syncthreads();
    nr = 0;
#pragma unroll
    for (int i = 0; i < NW; ++i) nr += red[i];
  }
  if (nr <= 0) return;
  // ---- this CTA's window (listed: 0, 1, last - 2, last - 1, last; clamped). Skipped: a repeat of an earlier entry, and a window the slide
  // does not move (its keys are the clipped window's: start 32 w - 48 >= 0 and the window inside the real atoms)
  const int last = (nr - 1) / WQ;
  int list[5] = {0, 1, last - 2, last - 1, last};
#pragma unroll
  for (int i = 0; i < 5; ++i) list[i] = min(max(list[i], 0), nwin - 1);
  const int e = blockIdx.x, w = list[e];
  for (int i = 0; i < e; ++i)
    if (list[i] == w) return;
  const int start = min(max(WQ * w - KOFF, 0), max(nr - WK, 0));
  if (start == WQ * w - KOFF && start + WK <= nr) return;
  const size_t base = (size_t)a * N;
  // ---- scores: thread = key j (its k row in registers) x one query half; q rows read as shared-memory broadcasts
  const int j = 32 * (warp & 3) + lane, qh = warp >> 2, key = start + j;   // query group qh: queries WQ / QG * qh ..
  const bool in = key < nr && key < N;
  float kr[DH];
  {
    const int r = (tid & 255) >> 1, c0 = 16 * (tid & 1), kv = start + r;   // threads 0..255: v row r; 256..319: q row r
    const bool vin = kv < nr && kv < N, isq = tid >= 256 && r < WQ, isv = tid < 256;
    float vv[16], qv[16];
    if (isv) load16(v + (base + (vin ? kv : 0)) * ld + DH * h + c0, vv);
    if (isq) load16(q + (base + WQ * w + r) * ld + DH * h + c0, qv);
#pragma unroll
    for (int c = 0; c < 16; c += 4) {
      if (isv) *reinterpret_cast<float4*>(&vs[r][c0 + c]) = vin ? make_float4(vv[c], vv[c + 1], vv[c + 2], vv[c + 3]) : make_float4(0.f, 0.f, 0.f, 0.f);
      if (isq) *reinterpret_cast<float4*>(&qs[r][c0 + c]) = make_float4(qv[c] * QSCALE, qv[c + 1] * QSCALE, qv[c + 2] * QSCALE, qv[c + 3] * QSCALE);
    }
    float k0[16], k1[16];
    load16(k + (base + (in ? key : 0)) * ld + DH * h, k0);
    load16(k + (base + (in ? key : 0)) * ld + DH * h + 16, k1);
#pragma unroll
    for (int c = 0; c < 16; ++c) { kr[c] = k0[c]; kr[16 + c] = k1[c]; }
  }
  const float* bw = bias + (((size_t)h * nwin + w) * WQ + WQ / QG * qh) * WK + j;
  float bq[WQ / QG];
#pragma unroll
  for (int i = 0; i < WQ / QG; ++i) bq[i] = __ldg(bw + i * WK);
  __syncthreads();
#pragma unroll
  for (int i = 0; i < WQ / QG; ++i) {
    const int r = WQ / QG * qh + i;
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int c4 = 0; c4 < DH; c4 += 4) {
      const float4 qv = *reinterpret_cast<const float4*>(&qs[r][c4]);
      acc[0] = fmaf(qv.x, kr[c4], acc[0]); acc[1] = fmaf(qv.y, kr[c4 + 1], acc[1]);
      acc[2] = fmaf(qv.z, kr[c4 + 2], acc[2]); acc[3] = fmaf(qv.w, kr[c4 + 3], acc[3]);
    }
    ss[r][j] = in ? ((acc[0] + acc[1]) + (acc[2] + acc[3])) + bq[i] : -INFINITY;
  }
  __syncthreads();
  // ---- softmax: a warp per 2 queries, P = exp(s - max) / sum written back over the scores
#pragma unroll
  for (int i = 0; i < WQ / NW; ++i) {
    const int r = WQ / NW * warp + i;
    float sv[4], mx = -INFINITY;
#pragma unroll
    for (int t = 0; t < 4; ++t) { sv[t] = ss[r][lane + 32 * t]; mx = fmaxf(mx, sv[t]); }
#pragma unroll
    for (int off = 16; off >= 1; off >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, off));
    float sum = 0.f;
#pragma unroll
    for (int t = 0; t < 4; ++t) { sv[t] = mx == -INFINITY ? 0.f : __expf(sv[t] - mx); sum += sv[t]; }
#pragma unroll
    for (int off = 16; off >= 1; off >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, off);
    const float inv = sum > 0.f ? 1.f / sum : 0.f;
#pragma unroll
    for (int t = 0; t < 4; ++t) ss[r][lane + 32 * t] = sv[t] * inv;
    if (lse != nullptr && lane == 0)
      lse[((size_t)a * NH + h) * N + WQ * w + r] = sum > 0.f ? (mx + logf(sum)) * LOG2E : -INFINITY;
  }
  __syncthreads();
  // ---- O = P V: thread = (query tid / 16, channels 2 (tid % 16) ..)
  {
    const int r = tid >> 4, c = 2 * (tid & 15);
    float acc[4] = {0.f, 0.f, 0.f, 0.f};                               // two channels x two key parities
#pragma unroll 16
    for (int jj = 0; jj < WK; jj += 2) {
      const float p0 = ss[r][jj], p1 = ss[r][jj + 1];
      const float2 v0 = *reinterpret_cast<const float2*>(&vs[jj][c]), v1 = *reinterpret_cast<const float2*>(&vs[jj + 1][c]);
      acc[0] = fmaf(p0, v0.x, acc[0]); acc[1] = fmaf(p0, v0.y, acc[1]); acc[2] = fmaf(p1, v1.x, acc[2]); acc[3] = fmaf(p1, v1.y, acc[3]);
    }
    T* op = o + (base + WQ * w + r) * DM + DH * h + c;
    st_f(op, acc[0] + acc[2]); st_f(op + 1, acc[1] + acc[3]);
  }
}

extern "C" __global__ void __launch_bounds__(512)
local_slide_edges(const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v, int ld, const float* bias, const uint8_t* km,
                  __nv_bfloat16* o, float* lse, int N, int n0) {
  slide_edges(q, k, v, ld, bias, km, o, lse, N, n0);
}

extern "C" __global__ void __launch_bounds__(512)
local_slide_edges_f32(const float* q, const float* k, const float* v, int ld, const float* bias, const uint8_t* km, float* o, float* lse,
                      int N, int n0) {
  slide_edges(q, k, v, ld, bias, km, o, lse, N, n0);
}
