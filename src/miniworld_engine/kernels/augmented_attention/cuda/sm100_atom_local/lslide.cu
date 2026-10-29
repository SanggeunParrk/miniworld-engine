// lslide.cu — the backward of AlphaFold 3's slid key window (``LocalDiTBlock(key_window="slide")``) around the clipped-window backward kernels.
//
// A slid window starts at S_w = min(max(32 w - 48, 0), max(n_real - 128, 0)) instead of 32 w - 48 (n_real: the real atoms, which come
// first; the trunked pair is laid out from the window's first key either way, so the per-window bias tensor is the same). Only the first
// two windows and the last three real windows can differ from the clipped window (a window ends at 32 w + 80). The forward runs them
// inside the attention kernels (slide.cuh); here the backward recomputes those windows over their slid keys. The window indices come from
// the real atom count on the device (the key mask's population count, or n0 without a mask): no host read, so the call is CUDA-graph
// capturable and a replay with another mask is right. A window listed twice is done once.
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

// ---- warp-level tensor-core products (mma.sync): C (16 x 8, fp32) += A (16 x K) B (K x 8) with A(m, k) = fa(m, k), B(k, n) = fb(k, n)
// read as fp32 values (shared memory) and rounded to the operand type: bf16 (m16n8k16) or tf32 (m16n8k8, to nearest as the fp32 path's
// kernels round every MMA operand). C fragment: c[i] = C[g + 8 (i >> 1)][2 t + (i & 1)], g = lane / 4, t = lane % 4.
DEVI uint32_t tf32r(float x) { return (__float_as_uint(x) + 0x1000u) & 0xffffe000u; }
DEVI uint32_t bf2(float lo, float hi) {
  const __nv_bfloat162 r = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<const uint32_t*>(&r);
}
template <bool F32, int K, typename FA, typename FB>
DEVI void tile_mma(float (&c)[4], FA fa, FB fb) {
  const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  if constexpr (F32) {
#pragma unroll
    for (int k0 = 0; k0 < K; k0 += 8) {
      const uint32_t a0 = tf32r(fa(g, k0 + t)), a1 = tf32r(fa(g + 8, k0 + t)), a2 = tf32r(fa(g, k0 + t + 4)), a3 = tf32r(fa(g + 8, k0 + t + 4));
      const uint32_t b0 = tf32r(fb(k0 + t, g)), b1 = tf32r(fb(k0 + t + 4, g));
      asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                   : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    }
  } else {
#pragma unroll
    for (int k0 = 0; k0 < K; k0 += 16) {
      const uint32_t a0 = bf2(fa(g, k0 + 2 * t), fa(g, k0 + 2 * t + 1)), a1 = bf2(fa(g + 8, k0 + 2 * t), fa(g + 8, k0 + 2 * t + 1));
      const uint32_t a2 = bf2(fa(g, k0 + 2 * t + 8), fa(g, k0 + 2 * t + 9)), a3 = bf2(fa(g + 8, k0 + 2 * t + 8), fa(g + 8, k0 + 2 * t + 9));
      const uint32_t b0 = bf2(fb(k0 + 2 * t, g), fb(k0 + 2 * t + 1, g)), b1 = bf2(fb(k0 + 2 * t + 8, g), fb(k0 + 2 * t + 9, g));
      asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                   : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    }
  }
}
// rows [r0, r0 + NROW) x the 32 channels of head h of a [*, ld] tensor into s[NROW][DH + 1] (fp32), rows r >= valid as zero: every 16-B
// load of a thread issued before any is stored (the latency paid once)
DEVI void unpack16(uint4 u, float* d, const float*) {
  d[0] = __uint_as_float(u.x); d[1] = __uint_as_float(u.y); d[2] = __uint_as_float(u.z); d[3] = __uint_as_float(u.w);
}
DEVI void unpack16(uint4 u, float* d, const __nv_bfloat16*) {
  const uint32_t w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
  for (int j = 0; j < 4; ++j) { d[2 * j] = __uint_as_float(w[j] << 16); d[2 * j + 1] = __uint_as_float(w[j] & 0xffff0000u); }
}
DEVI uint4 pack16(const float* x, const float*) {
  return make_uint4(__float_as_uint(x[0]), __float_as_uint(x[1]), __float_as_uint(x[2]), __float_as_uint(x[3]));
}
DEVI uint4 pack16(const float* x, const __nv_bfloat16*) { return make_uint4(bf2(x[0], x[1]), bf2(x[2], x[3]), bf2(x[4], x[5]), bf2(x[6], x[7])); }
template <int NROW, int NT, typename T>
DEVI void stage_rows(float (*s)[DH + 1], const T* __restrict__ x, size_t r0, int ld, int h, int valid) {
  constexpr int EPV = 16 / (int)sizeof(T), VPR = DH / EPV, NV = NROW * VPR, PER = (NV + NT - 1) / NT;
  uint4 buf[PER];
#pragma unroll
  for (int p = 0; p < PER; ++p) {
    const int idx = (int)threadIdx.x + p * NT, r = idx / VPR, cv = idx % VPR;
    buf[p] = (idx < NV && r < valid) ? __ldg(reinterpret_cast<const uint4*>(x + (r0 + r) * ld + DH * h + cv * EPV)) : make_uint4(0u, 0u, 0u, 0u);
  }
#pragma unroll
  for (int p = 0; p < PER; ++p) {
    const int idx = (int)threadIdx.x + p * NT, r = idx / VPR, cv = idx % VPR;
    if (idx < NV) {
      float d[EPV];
      unpack16(buf[p], d, x);
#pragma unroll
      for (int e = 0; e < EPV; ++e) s[r][cv * EPV + e] = d[e];
    }
  }
}

// the window's bias [32][128] (fp32, natural units) into s[32][BST], 16-B loads all issued first
constexpr int BST = WK + 4;
template <int NT>
DEVI void stage_bias(float (*s)[BST], const float* __restrict__ bw) {
  constexpr int NV = WQ * WK / 4, PER = NV / NT;
  float4 buf[PER];
#pragma unroll
  for (int p = 0; p < PER; ++p) buf[p] = __ldg(reinterpret_cast<const float4*>(bw) + threadIdx.x + p * NT);
#pragma unroll
  for (int p = 0; p < PER; ++p) {
    const int idx = (int)threadIdx.x + p * NT, r = idx / (WK / 4), c = 4 * (idx % (WK / 4));
    *reinterpret_cast<float4*>(&s[r][c]) = buf[p];
  }
}

// =================================================================================================================== training (backward)
// The slid windows' gradients, around the clipped-window backward kernels (the forward wrote O and
// the LSE of the moved windows over their slid keys):
//   local_slide_bwd_prep   grid (5, A): the moved windows' rows of dO [A N, 128] and D [A, 4, N] are copied to sdo [5][A][32][128] /
//                          sdd [5][A][4][32] and zeroed in place -- the clipped-window kernels then give those queries dS = P (dP - D) = 0
//                          (no dQ, dK, dV or dbias from them)
//   local_slide_bwd        grid (2, 4, A), 128 threads: one key group (the moved windows sharing their slid keys), head and sample on
//                          tensor cores: P = 2^(s - LSE), dP = dO V^T, dS = P (dP - D); dQ written into dq; dK = dS^T Q / sqrt(32) and
//                          dV = P^T dO summed over the group's windows and added into dk / dv (one writer per key); dS as the dbias
//                          partial sdb [5][A][4][32][128]
//   local_slide_bwd_fin    dbias of each moved window = the sum over the samples of its partials, in order (deterministic)
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


// local_slide_bwd: one CTA (4 warps) per (key group, head, sample). The moved windows of a group share their slid keys -- group 0: entries
// 0, 1 (windows 0, 1, keys [0, 128)); group 1: entries 2..4 (the last three, keys [max(n_real - 128, 0), ..)) -- so the CTA sums their dK / dV
// in its MMA accumulators and adds them into dk / dv itself (the clipped-window kernels have written those keys' other contributions). When
// the two groups' keys overlap (n_real < 256) group 0's CTA does both groups in order (group 1's exits): one writer per key, deterministic.
// Per window, on tensor cores: S = Q K^T, dP = dO V^T (P = 2^(S c + bias - LSE), dS = P (dP - D) on the fragments), dQ = dS K / sqrt(32)
// written into dq, dK += dS^T Q / sqrt(32), dV += P^T dO; dS is the window's dbias partial sdb [5][A][4][32][128] for the sample sum.
constexpr int BWD_THREADS = 128;
constexpr int SLIDE_BWD_SMEM = ((2 * WQ + 2 * WK) * (DH + 1) + 2 * WQ * (WK + 1) + 2 * WQ + WQ * BST) * 4;   // 92 KB
static_assert(SLIDE_BWD_SMEM == 92416, "keep sm100_atom_local.KERNELS['slide_b'] in step");

template <typename T>
DEVI void slide_bwd(const T* __restrict__ q, const T* __restrict__ k, const T* __restrict__ v, int ld, const float* __restrict__ bias,
                    const uint8_t* __restrict__ km, const float* __restrict__ lse, const T* __restrict__ sdo, const float* __restrict__ sdd,
                    T* __restrict__ dq, T* __restrict__ dk, T* __restrict__ dv, int ldd, float* __restrict__ sdb, int N, int n0, int A) {
  constexpr bool F32 = sizeof(T) == 4;
  extern __shared__ __align__(16) float smf[];                         // dynamic: SLIDE_BWD_SMEM bytes
  float(*qs)[DH + 1] = reinterpret_cast<float(*)[DH + 1]>(smf);
  float(*ks)[DH + 1] = reinterpret_cast<float(*)[DH + 1]>(smf + WQ * (DH + 1));
  float(*vs)[DH + 1] = reinterpret_cast<float(*)[DH + 1]>(smf + (WQ + WK) * (DH + 1));
  float(*os)[DH + 1] = reinterpret_cast<float(*)[DH + 1]>(smf + (WQ + 2 * WK) * (DH + 1));
  float(*ps)[WK + 1] = reinterpret_cast<float(*)[WK + 1]>(smf + (2 * WQ + 2 * WK) * (DH + 1));
  float(*ds)[WK + 1] = reinterpret_cast<float(*)[WK + 1]>(smf + (2 * WQ + 2 * WK) * (DH + 1) + WQ * (WK + 1));
  float* lrow = smf + (2 * WQ + 2 * WK) * (DH + 1) + 2 * WQ * (WK + 1);
  float* drow = lrow + WQ;
  float(*bs)[BST] = reinterpret_cast<float(*)[BST]>(drow + WQ);
  __shared__ int red[BWD_THREADS / 32];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const int h = blockIdx.y, a = blockIdx.z, nwin = N / WQ;
  const int nr = real_count(km, N, n0, red);
  if (nr <= 0) return;
  const bool overlap = nr < 2 * WK;
  if (overlap && blockIdx.x == 1) return;
  const size_t base = (size_t)a * N;
  for (int grp = blockIdx.x; grp <= (overlap ? 1 : (int)blockIdx.x); ++grp) {
    const int start = grp == 0 ? 0 : max(nr - WK, 0), kvalid = min(nr, N) - start;
    bool any = false;
#pragma unroll 1
    for (int e = 2 * grp; e < (grp == 0 ? 2 : 5); ++e) {
      int w, st;
      any = any || edge_window(e, nr, nwin, w, st);
    }
    if (!any) continue;
    __syncthreads();                                                   // (the previous group's reads of K / V)
    stage_rows<WK, BWD_THREADS>(ks, k, base + start, ld, h, kvalid);
    stage_rows<WK, BWD_THREADS>(vs, v, base + start, ld, h, kvalid);
    float acc[2][2][DH / 8][4];                                        // [dK | dV][key tile][channel tile]: keys 32 warp + 16 mt ..
#pragma unroll
    for (int x = 0; x < 2; ++x)
#pragma unroll
      for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int nt = 0; nt < DH / 8; ++nt) acc[x][mt][nt][0] = acc[x][mt][nt][1] = acc[x][mt][nt][2] = acc[x][mt][nt][3] = 0.f;
#pragma unroll 1
    for (int e = 2 * grp; e < (grp == 0 ? 2 : 5); ++e) {
      int w, st;
      if (!edge_window(e, nr, nwin, w, st)) continue;                  // (st == start: the group's slid keys)
      __syncthreads();                                                 // (the previous window's reads of Q, dO, P, dS)
      const float* bw = bias + (((size_t)h * nwin + w) * WQ) * WK;
      stage_bias<BWD_THREADS>(bs, bw);
      stage_rows<WQ, BWD_THREADS>(qs, q, base + WQ * w, ld, h, WQ);
      stage_rows<WQ, BWD_THREADS>(os, sdo, ((size_t)e * A + a) * WQ, DM, h, WQ);
      if (tid < WQ) {
        lrow[tid] = lse[((size_t)a * NH + h) * N + WQ * w + tid];
        drow[tid] = sdd[(((size_t)e * A + a) * NH + h) * WQ + tid];
      }
      __syncthreads();
      // S, dP for keys 32 warp .. (4 tiles of 8) x the 32 queries (2 tiles of 16); P, dS on the fragments
      float* dbp = sdb + (((size_t)e * A + a) * NH + h) * WQ * WK;
      float sc[2][4][4], pc[2][4][4];                                  // all tiles first: each operand fragment is loaded once
#pragma unroll
      for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
          const int j0 = 32 * warp + 8 * nt;
#pragma unroll
          for (int i = 0; i < 4; ++i) sc[mt][nt][i] = pc[mt][nt][i] = 0.f;
          tile_mma<F32, DH>(sc[mt][nt], [&](int m, int kk) { return qs[16 * mt + m][kk]; }, [&](int kk, int n) { return ks[j0 + n][kk]; });
          tile_mma<F32, DH>(pc[mt][nt], [&](int m, int kk) { return os[16 * mt + m][kk]; }, [&](int kk, int n) { return vs[j0 + n][kk]; });
        }
#pragma unroll
      for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int nt = 0; nt < 4; ++nt)
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            const int r = 16 * mt + g + 8 * (i >> 1), j = 32 * warp + 8 * nt + 2 * t + (i & 1);
            const float p = j < kvalid ? exp2f(fmaf(sc[mt][nt][i], QSCALE, bs[r][j]) * LOG2E - lrow[r]) : 0.f;
            const float d = p * (pc[mt][nt][i] - drow[r]);
            ps[r][j] = p;
            ds[r][j] = d;
          }
      __syncthreads();
      for (int i = tid; i < WQ * WK / 4; i += BWD_THREADS) {          // dS -> the window's dbias partial, 16-B stores
        const int r = i / (WK / 4), c = 4 * (i % (WK / 4));
        *reinterpret_cast<float4*>(dbp + r * WK + c) = make_float4(ds[r][c], ds[r][c + 1], ds[r][c + 2], ds[r][c + 3]);
      }
      // dQ = dS K / sqrt(32): warp -> queries 16 (warp >> 1) .., channels 16 (warp & 1) ..
      {
        const int m0 = 16 * (warp >> 1);
        float qc[2][4];
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
          const int n0c = 16 * (warp & 1) + 8 * nt;
          qc[nt][0] = qc[nt][1] = qc[nt][2] = qc[nt][3] = 0.f;
          tile_mma<F32, WK>(qc[nt], [&](int m, int kk) { return ds[m0 + m][kk]; }, [&](int kk, int n) { return ks[kk][n0c + n]; });
        }
#pragma unroll
        for (int nt = 0; nt < 2; ++nt)
#pragma unroll
          for (int x = 0; x < 2; ++x) {
            T* out = dq + (base + WQ * w + m0 + g + 8 * x) * ldd + DH * h + 16 * (warp & 1) + 8 * nt + 2 * t;
            st_f(out, qc[nt][2 * x] * QSCALE); st_f(out + 1, qc[nt][2 * x + 1] * QSCALE);
          }
      }
      // dK += dS^T Q, dV += P^T dO for keys 32 warp ..
#pragma unroll
      for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int nt = 0; nt < DH / 8; ++nt) {
          const int j0 = 32 * warp + 16 * mt;
          tile_mma<F32, WQ>(acc[0][mt][nt], [&](int m, int kk) { return ds[kk][j0 + m]; }, [&](int kk, int n) { return qs[kk][8 * nt + n]; });
          tile_mma<F32, WQ>(acc[1][mt][nt], [&](int m, int kk) { return ps[kk][j0 + m]; }, [&](int kk, int n) { return os[kk][8 * nt + n]; });
        }
    }
    // dk / dv += the group's sums at its real keys: the sums through shared memory (over P / dS), then 16-B read-modify-writes
    __syncthreads();                                                   // (every warp's last reads of P / dS)
    float(*sacc)[2 * DH + 1] = reinterpret_cast<float(*)[2 * DH + 1]>(&ps[0][0]);   // [key][dK | dV] over P, dS, LSE, D
    static_assert(WK * (2 * DH + 1) <= 2 * WQ * (WK + 1) + 2 * WQ, "the sums over P, dS, LSE and D (contiguous)");
#pragma unroll
    for (int x = 0; x < 2; ++x)
#pragma unroll
      for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int nt = 0; nt < DH / 8; ++nt)
#pragma unroll
          for (int i = 0; i < 4; ++i)
            sacc[32 * warp + 16 * mt + g + 8 * (i >> 1)][DH * x + 8 * nt + 2 * t + (i & 1)] = acc[x][mt][nt][i] * (x == 0 ? QSCALE : 1.f);
    __syncthreads();
    {
      constexpr int EPV = 16 / (int)sizeof(T), VPR = 2 * DH / EPV, NV = WK * VPR, PER = NV / BWD_THREADS;   // (key, dK | dV vector)
      uint4 cur[PER];
#pragma unroll
      for (int p = 0; p < PER; ++p) {
        const int idx = tid + p * BWD_THREADS, j = idx / VPR, cv = idx % VPR, c = cv * EPV;
        T* dst = (c < DH ? dk : dv) + (base + start + j) * ldd + DH * h + (c % DH);
        cur[p] = j < kvalid ? *reinterpret_cast<const uint4*>(dst) : make_uint4(0u, 0u, 0u, 0u);
      }
#pragma unroll
      for (int p = 0; p < PER; ++p) {
        const int idx = tid + p * BWD_THREADS, j = idx / VPR, cv = idx % VPR, c = cv * EPV;
        if (j >= kvalid) continue;
        T* dst = (c < DH ? dk : dv) + (base + start + j) * ldd + DH * h + (c % DH);
        float x[EPV];
        unpack16(cur[p], x, dst);
#pragma unroll
        for (int e = 0; e < EPV; ++e) x[e] += sacc[j][c + e];
        *reinterpret_cast<uint4*>(dst) = pack16(x, dst);
      }
    }
  }
}

// local_slide_bwd_fin: dbias of each moved window = the sum over the samples of its partials, in the samples' order (deterministic)
DEVI void slide_bwd_fin(const uint8_t* __restrict__ km, const float* __restrict__ sdb, float* __restrict__ dbias, int N, int n0, int A) {
  __shared__ int red[8];
  const int nwin = N / WQ;
  const int nr = real_count(km, N, n0, red);
  int ws[5], st[5];
  bool act[5];
#pragma unroll
  for (int e = 0; e < 5; ++e) act[e] = edge_window(e, nr, nwin, ws[e], st[e]);
  const int ndb = 5 * NH * WQ * WK;
  for (int y = blockIdx.x * blockDim.x + threadIdx.x; y < ndb; y += gridDim.x * blockDim.x) {   // (e, h, i, j)
    const int ij = y % (WQ * WK), hh = (y / (WQ * WK)) % NH, e = y / (WQ * WK * NH);
    bool ae = false;
    int we = 0;
#pragma unroll
    for (int f = 0; f < 5; ++f)                                      // (selected with constant indices: no local-memory arrays)
      if (f == e) { ae = act[f]; we = ws[f]; }
    if (!ae) continue;
    const float* src = sdb + ((size_t)e * A * NH + hh) * WQ * WK + ij;
    float sum = 0.f;
    for (int a = 0; a < A; ++a) sum += __ldg(src + (size_t)a * NH * WQ * WK);
    dbias[((size_t)hh * nwin + we) * WQ * WK + ij] = sum;
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
extern "C" __global__ void __launch_bounds__(BWD_THREADS, 3)
local_slide_bwd(const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v, int ld, const float* bias, const uint8_t* km,
                const float* lse, const __nv_bfloat16* sdo, const float* sdd, __nv_bfloat16* dq, __nv_bfloat16* dk, __nv_bfloat16* dv, int ldd,
                float* sdb, int N, int n0, int A) {
  slide_bwd(q, k, v, ld, bias, km, lse, sdo, sdd, dq, dk, dv, ldd, sdb, N, n0, A);
}
extern "C" __global__ void __launch_bounds__(BWD_THREADS, 3)
local_slide_bwd_f32(const float* q, const float* k, const float* v, int ld, const float* bias, const uint8_t* km, const float* lse,
                    const float* sdo, const float* sdd, float* dq, float* dk, float* dv, int ldd, float* sdb, int N, int n0, int A) {
  slide_bwd(q, k, v, ld, bias, km, lse, sdo, sdd, dq, dk, dv, ldd, sdb, N, n0, A);
}
extern "C" __global__ void __launch_bounds__(256)
local_slide_bwd_fin(const uint8_t* km, const float* sdb, float* dbias, int N, int n0, int A) {
  slide_bwd_fin(km, sdb, dbias, N, n0, A);
}
