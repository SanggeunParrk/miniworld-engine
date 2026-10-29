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
DEVI float ld_f(const float* p) { return *p; }
DEVI float ld_f(const __nv_bfloat16* p) { return __bfloat162float(*p); }

// the real atom count: the key mask's population count (every thread of the block takes part; red: one int per warp), or n0 without a mask
DEVI int real_count(const uint8_t* __restrict__ km, int N, int n0, int* red) {
  if (km == nullptr) return n0;
  const int tid = threadIdx.x, lane = tid & 31, nw = blockDim.x >> 5;
  int c = 0;
  for (int i = 16 * tid; i < N; i += 16 * blockDim.x) {                // N is a multiple of 128: whole 16-byte groups
    const uint4 b = *reinterpret_cast<const uint4*>(km + i);
    const uint32_t w4[4] = {b.x, b.y, b.z, b.w};
#pragma unroll
    for (int j = 0; j < 4; ++j) c += ((w4[j] & 0xffu) != 0) + ((w4[j] & 0xff00u) != 0) + ((w4[j] & 0xff0000u) != 0) + ((w4[j] >> 24) != 0);
  }
#pragma unroll
  for (int off = 16; off >= 1; off >>= 1) c += __shfl_xor_sync(0xffffffffu, c, off);
  if (lane == 0) red[tid >> 5] = c;
  __syncthreads();
  int nr = 0;
  for (int i = 0; i < nw; ++i) nr += red[i];
  __syncthreads();
  return nr;
}
// listed window e (0, 1, last - 2, last - 1, last; clamped): its index and slid start; false when it is a repeat of an earlier entry or
// the slide does not move it (its keys are the clipped window's)
DEVI int listed(int e, int last, int nwin) {                           // entry e of (0, 1, last - 2, last - 1, last), clamped
  const int w = e == 0 ? 0 : e == 1 ? 1 : last - 4 + e;
  return min(max(w, 0), nwin - 1);
}
DEVI bool edge_window(int e, int nr, int nwin, int& w, int& start) {
  if (nr <= 0) return false;
  const int last = (nr - 1) / WQ;
  w = listed(e, last, nwin);
#pragma unroll
  for (int i = 0; i < 4; ++i)
    if (i < e && listed(i, last, nwin) == w) return false;
  start = min(max(WQ * w - KOFF, 0), max(nr - WK, 0));
  return !(start == WQ * w - KOFF && start + WK <= nr);
}

template <typename T>
DEVI void slide_edges(const T* __restrict__ q, const T* __restrict__ k, const T* __restrict__ v, int ld, const float* __restrict__ bias,
                      const uint8_t* __restrict__ km, T* __restrict__ o, float* __restrict__ lse, int N, int n0) {
  __shared__ __align__(16) float qs[WQ][DH], vs[WK][DH], ss[WQ][WK + 1];   // 45.1 KB (under the 48 KB static limit)
  __shared__ int red[NW];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int h = blockIdx.y, a = blockIdx.z, nwin = N / WQ;
  const int nr = real_count(km, N, n0, red);
  int w, start;
  if (!edge_window(blockIdx.x, nr, nwin, w, start)) return;
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

// =================================================================================================================== training (backward)
// The slid windows' gradients, around the clipped-window backward kernels (the forward with ``save`` ran slide_edges with the LSE, so O and
// the LSE of the moved windows are right):
//   local_slide_bwd_prep   grid (5, A): the moved windows' rows of dO [A N, 128] and D [A, 4, N] are copied to sdo [5][A][32][128] /
//                          sdd [5][A][4][32] and zeroed in place -- the clipped-window kernels then give those queries dS = P (dP - D) = 0
//                          (no dQ, dK, dV or dbias from them)
//   local_slide_bwd        grid (5, 4, A), 256 threads: one moved window, head and sample over its slid keys: P = 2^(s - LSE), dP = dO V^T,
//                          dS = P (dP - D); dQ (rows of the window) written into dq; dK = dS^T Q / sqrt(32), dV = P^T dO and dbias = dS as
//                          fp32 partials skv [5][A][128][4][64] (k | v per head) / sdb [5][A][4][32][128]
//   local_slide_bwd_fin    the partials summed in a fixed order (deterministic): dK / dV added into dk / dv at each slid key (the entries
//                          that contain the key, in list order, by the first of them), dbias of each moved window = the sum over the samples
// q, k, v, dq, dk, dv: [A, N, *] views (rows ld / ldd elements apart); o, do [A N, 128]; lse, D [A, 4, N].
template <typename T>
DEVI void slide_bwd_prep(T* __restrict__ dout, float* __restrict__ D, const uint8_t* __restrict__ km, T* __restrict__ sdo,
                         float* __restrict__ sdd, int N, int n0, int A) {
  __shared__ int red[8];
  const int tid = threadIdx.x, e = blockIdx.x, a = blockIdx.y, nwin = N / WQ;
  const int nr = real_count(km, N, n0, red);
  int w, start;
  if (!edge_window(e, nr, nwin, w, start)) return;
  T* rows = dout + ((size_t)a * N + WQ * w) * DM;
  T* dst = sdo + ((size_t)e * A + a) * WQ * DM;
  for (int i = tid; i < WQ * DM; i += blockDim.x) { dst[i] = rows[i]; st_f(rows + i, 0.f); }
  if (tid < NH * WQ) {
    const int hh = tid / WQ, r = tid % WQ;
    float* dp = D + ((size_t)a * NH + hh) * N + WQ * w + r;
    sdd[(((size_t)e * A + a) * NH + hh) * WQ + r] = *dp;
    *dp = 0.f;
  }
}

constexpr int SLIDE_BWD_SMEM = ((2 * WQ + 2 * WK) * (DH + 1) + 2 * WQ * (WK + 1) + 2 * WQ) * 4;   // 75 KB
static_assert(SLIDE_BWD_SMEM == 75520, "keep sm100_atom_local.KERNELS['slide_b'] in step");

template <typename T>
DEVI void slide_bwd(const T* __restrict__ q, const T* __restrict__ k, const T* __restrict__ v, int ld, const float* __restrict__ bias,
                    const uint8_t* __restrict__ km, const float* __restrict__ lse, const T* __restrict__ sdo, const float* __restrict__ sdd,
                    T* __restrict__ dq, int ldd, float* __restrict__ skv, float* __restrict__ sdb, int N, int n0, int A) {
  extern __shared__ __align__(16) float smf[];                         // dynamic: SLIDE_BWD_SMEM bytes
  float(*qs)[DH + 1] = reinterpret_cast<float(*)[DH + 1]>(smf);
  float(*ks)[DH + 1] = reinterpret_cast<float(*)[DH + 1]>(smf + WQ * (DH + 1));
  float(*vs)[DH + 1] = reinterpret_cast<float(*)[DH + 1]>(smf + (WQ + WK) * (DH + 1));
  float(*os)[DH + 1] = reinterpret_cast<float(*)[DH + 1]>(smf + (WQ + 2 * WK) * (DH + 1));
  float(*ps)[WK + 1] = reinterpret_cast<float(*)[WK + 1]>(smf + (2 * WQ + 2 * WK) * (DH + 1));
  float(*ds)[WK + 1] = reinterpret_cast<float(*)[WK + 1]>(smf + (2 * WQ + 2 * WK) * (DH + 1) + WQ * (WK + 1));
  float* lrow = smf + (2 * WQ + 2 * WK) * (DH + 1) + 2 * WQ * (WK + 1);
  float* drow = lrow + WQ;
  __shared__ int red[8];
  const int tid = threadIdx.x, e = blockIdx.x, h = blockIdx.y, a = blockIdx.z, nwin = N / WQ;
  const int nr = real_count(km, N, n0, red);
  int w, start;
  if (!edge_window(e, nr, nwin, w, start)) return;
  const size_t base = (size_t)a * N;
  for (int i = tid; i < WK * DH; i += blockDim.x) {
    const int r = i >> 5, c = i & 31, key = start + r;
    const bool in = key < nr && key < N;
    ks[r][c] = in ? ld_f(k + (base + key) * ld + DH * h + c) : 0.f;
    vs[r][c] = in ? ld_f(v + (base + key) * ld + DH * h + c) : 0.f;
    if (r < WQ) {
      qs[r][c] = ld_f(q + (base + WQ * w + r) * ld + DH * h + c);
      os[r][c] = ld_f(sdo + (((size_t)e * A + a) * WQ + r) * DM + DH * h + c);   // dO
    }
  }
  if (tid < WQ) {
    lrow[tid] = lse[((size_t)a * NH + h) * N + WQ * w + tid];
    drow[tid] = sdd[(((size_t)e * A + a) * NH + h) * WQ + tid];
  }
  __syncthreads();
  const float* bw = bias + (((size_t)h * nwin + w) * WQ) * WK;
  // P and dS: thread (query i = tid / 8, keys j = tid % 8 + 8 m)
  {
    const int i = tid >> 3, j0 = tid & 7;
    float qr[DH], dor[DH];
#pragma unroll
    for (int c = 0; c < DH; ++c) { qr[c] = qs[i][c]; dor[c] = os[i][c]; }
    const float li = lrow[i], di = drow[i];
    for (int m = 0; m < WK / 8; ++m) {
      const int j = j0 + 8 * m, key = start + j;
      float s = 0.f, dp = 0.f;
#pragma unroll
      for (int c = 0; c < DH; ++c) { s = fmaf(qr[c], ks[j][c], s); dp = fmaf(dor[c], vs[j][c], dp); }
      const bool in = key < nr && key < N;
      const float p = in ? exp2f((s * QSCALE + bw[i * WK + j]) * LOG2E - li) : 0.f;
      ps[i][j] = p;
      ds[i][j] = p * (dp - di);
    }
  }
  __syncthreads();
  float* dbp = sdb + (((size_t)e * A + a) * NH + h) * WQ * WK;
  for (int i = tid; i < WQ * WK; i += blockDim.x) dbp[i] = ds[i >> 7][i & 127];
  // dQ = dS K / sqrt(32): thread (query tid / 8, channels 4 (tid % 8) ..)
  {
    const int i = tid >> 3, c0 = 4 * (tid & 7);
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    for (int j = 0; j < WK; ++j) {
      const float d = ds[i][j];
#pragma unroll
      for (int c = 0; c < 4; ++c) acc[c] = fmaf(d, ks[j][c0 + c], acc[c]);
    }
    T* out = dq + (base + WQ * w + i) * ldd + DH * h + c0;
#pragma unroll
    for (int c = 0; c < 4; ++c) st_f(out + c, acc[c] * QSCALE);
  }
  // dK = dS^T Q / sqrt(32), dV = P^T dO: thread (key tid / 2, channels 16 (tid % 2) ..)
  {
    const int j = tid >> 1, c0 = 16 * (tid & 1);
    float ak[16], av[16];
#pragma unroll
    for (int c = 0; c < 16; ++c) { ak[c] = 0.f; av[c] = 0.f; }
    for (int i = 0; i < WQ; ++i) {
      const float d = ds[i][j], p = ps[i][j];
#pragma unroll
      for (int c = 0; c < 16; ++c) { ak[c] = fmaf(d, qs[i][c0 + c], ak[c]); av[c] = fmaf(p, os[i][c0 + c], av[c]); }
    }
    float* dst = skv + ((((size_t)e * A + a) * WK + j) * NH + h) * 64;
#pragma unroll
    for (int c = 0; c < 16; ++c) { dst[c0 + c] = ak[c] * QSCALE; dst[32 + c0 + c] = av[c]; }
  }
}

template <typename T>
DEVI void slide_bwd_fin(const uint8_t* __restrict__ km, const float* __restrict__ skv, const float* __restrict__ sdb, T* __restrict__ dk,
                        T* __restrict__ dv, int ldd, float* __restrict__ dbias, int N, int n0, int A) {
  __shared__ int red[8];
  const int nwin = N / WQ;
  const int nr = real_count(km, N, n0, red);
  int ws[5], st[5];
  bool act[5];
#pragma unroll
  for (int e = 0; e < 5; ++e) act[e] = edge_window(e, nr, nwin, ws[e], st[e]);
  const size_t nkv = (size_t)5 * A * WK * NH * 64, ndb = (size_t)5 * NH * WQ * WK;
  for (size_t x = (size_t)blockIdx.x * blockDim.x + threadIdx.x; x < nkv + ndb; x += (size_t)gridDim.x * blockDim.x) {
    if (x < nkv) {                                                   // (e, a, j, h, c): key p = start_e + j, by the first entry holding p
      const int c = (int)(x % 64), hh = (int)((x / 64) % NH), j = (int)((x / (64 * NH)) % WK);
      const int a = (int)((x / ((size_t)64 * NH * WK)) % A), e = (int)(x / ((size_t)64 * NH * WK * A));
      bool ae = false;
      int se = 0;
#pragma unroll
      for (int f = 0; f < 5; ++f)                                    // (selected with constant indices: no local-memory arrays)
        if (f == e) { ae = act[f]; se = st[f]; }
      if (!ae) continue;
      const int p = se + j;
      if (p >= nr || p >= N) continue;
      bool first = true;
#pragma unroll
      for (int f = 0; f < 5; ++f) first = first && !(f < e && act[f] && p >= st[f] && p < st[f] + WK);
      if (!first) continue;
      float sum = 0.f;
#pragma unroll
      for (int f = 0; f < 5; ++f)
        if (act[f] && p >= st[f] && p < st[f] + WK) sum += skv[((((size_t)f * A + a) * WK + (p - st[f])) * NH + hh) * 64 + c];
      T* dst = (c < 32 ? dk : dv) + ((size_t)a * N + p) * ldd + DH * hh + (c & 31);
      st_f(dst, ld_f(dst) + sum);
    } else {                                                         // (e, h, i, j): dbias of the moved window, summed over the samples
      const size_t y = x - nkv;
      const int ij = (int)(y % (WQ * WK)), hh = (int)((y / (WQ * WK)) % NH), e = (int)(y / ((size_t)WQ * WK * NH));
      bool ae = false;
      int we = 0;
#pragma unroll
      for (int f = 0; f < 5; ++f)
        if (f == e) { ae = act[f]; we = ws[f]; }
      if (!ae) continue;
      float sum = 0.f;
      for (int a = 0; a < A; ++a) sum += sdb[(((size_t)e * A + a) * NH + hh) * WQ * WK + ij];
      dbias[((size_t)hh * nwin + we) * WQ * WK + ij] = sum;
    }
  }
}

extern "C" __global__ void __launch_bounds__(256)
local_slide_bwd_prep(__nv_bfloat16* dout, float* D, const uint8_t* km, __nv_bfloat16* sdo, float* sdd, int N, int n0, int A) {
  slide_bwd_prep(dout, D, km, sdo, sdd, N, n0, A);
}
extern "C" __global__ void __launch_bounds__(256)
local_slide_bwd_prep_f32(float* dout, float* D, const uint8_t* km, float* sdo, float* sdd, int N, int n0, int A) {
  slide_bwd_prep(dout, D, km, sdo, sdd, N, n0, A);
}
extern "C" __global__ void __launch_bounds__(256)
local_slide_bwd(const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v, int ld, const float* bias, const uint8_t* km,
                const float* lse, const __nv_bfloat16* sdo, const float* sdd, __nv_bfloat16* dq, int ldd, float* skv, float* sdb, int N, int n0,
                int A) {
  slide_bwd(q, k, v, ld, bias, km, lse, sdo, sdd, dq, ldd, skv, sdb, N, n0, A);
}
extern "C" __global__ void __launch_bounds__(256)
local_slide_bwd_f32(const float* q, const float* k, const float* v, int ld, const float* bias, const uint8_t* km, const float* lse,
                    const float* sdo, const float* sdd, float* dq, int ldd, float* skv, float* sdb, int N, int n0, int A) {
  slide_bwd(q, k, v, ld, bias, km, lse, sdo, sdd, dq, ldd, skv, sdb, N, n0, A);
}
extern "C" __global__ void __launch_bounds__(256)
local_slide_bwd_fin(const uint8_t* km, const float* skv, const float* sdb, __nv_bfloat16* dk, __nv_bfloat16* dv, int ldd, float* dbias, int N,
                    int n0, int A) {
  slide_bwd_fin(km, skv, sdb, dk, dv, ldd, dbias, N, n0, A);
}
extern "C" __global__ void __launch_bounds__(256)
local_slide_bwd_fin_f32(const uint8_t* km, const float* skv, const float* sdb, float* dk, float* dv, int ldd, float* dbias, int N, int n0,
                        int A) {
  slide_bwd_fin(km, skv, sdb, dk, dv, ldd, dbias, N, n0, A);
}
