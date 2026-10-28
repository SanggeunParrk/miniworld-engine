// AttentionPairBias fp32 (TF32) on H100: the row / pair kernels around the TF32 wgmma attention core (attn_tf32.cu).
//
//   forward   apb_prep      q (+ bias, x log2 e / sqrt d, TF32-rounded), k, v rounded, v^T -- the projections' packed [L, 4N] output unpacked
//                           into the core's layout (head width padded to DH: zero columns, so a 24-wide head runs as 32)
//             pb_fwd        pair -> bias [NH, L, L]: LayerNorm + the head projection on mma.sync TF32 (one read of the pair), written in
//                           the core's key order with the key mask folded in (log2 units)
//             apb_gate      og = sigmoid(g) O, the padding dropped
//   backward  apb_bwd_prep  dO = d og sigmoid(g) (rounded for the core), dg, D = rowsum(dO O), the LSE / D copies in the core's query order
//             apb_t3        q^T, k^T, dO^T (the core's transposed operands)  /  apb_bias_t   the bias transposed
//             apb_post      dq (rescaled to the real head width), dk, dv -> the packed [L, 4N] gradient of the projections
//             pb_bwd_tc     dpair and the dWf block partials, d_pair 128 (tensor cores; natural dbias)
//             pb_bwd_rows   the same for d_pair 256 / 8 heads on the FMA pipe (warp per row)
//
// One batch row (A = 1 sample): the layouts below are [L, ...] row-major fp32.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <algorithm>
#include <cstdint>
#include <vector>

#include "../../conditioned_transition/cuda/token_dit_common.cuh"  // tdr::warp_sum

namespace {
constexpr float LOG2E_F = 1.4426950408889634f;

// key_perm (conditioned_transition.cuda): position p of each group of 8 holds key perm(p); inv is its inverse
__device__ __forceinline__ int kperm(int p) { const int r = p & 7; return p - r + 4 * (r & 1) + (r >> 1); }
__device__ __forceinline__ int kinv(int k) { const int s = k & 7; return k - s + 2 * (s & 3) + (s >> 2); }
// TF32 round to nearest (ties away from zero) of a finite float: what cuBLAS does to an fp32 operand
__device__ __forceinline__ float rna_tf32(float x) { return __uint_as_float((__float_as_uint(x) + 0x1000u) & 0xffffe000u); }
__device__ __forceinline__ float sigf(float v) { return 1.f / (1.f + expf(-v)); }
__device__ __forceinline__ float4 ld4(const float* p) { return *reinterpret_cast<const float4*>(p); }
__device__ __forceinline__ float sum4(float4 a) { return (a.x + a.y) + (a.z + a.w); }
__device__ __forceinline__ float ldsv(const float* p) {   // a shared-memory read the compiler may not hoist (bounds registers)
  float v;
  asm volatile("ld.shared.f32 %0, [%1];" : "=f"(v) : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(p))));
  return v;
}

// ------------------------------------------------------------------------------------------------------------- row kernels
// QKVG [L, 4N] (q | k | v | g, N = NH * DR) -> QS = rna((q + bq) scale), KR = rna(k), QF = q + bq (training), VT = rna(v)^T, all padded to DH per head
template <int NH, int DR, int DH>
__global__ void __launch_bounds__(256) prep_k(const float* __restrict__ QKVG, const float* __restrict__ BQ, float* __restrict__ QS,
                                              float* __restrict__ KR, float* __restrict__ QF, float* __restrict__ VT, int L, float scale) {
  constexpr int N = NH * DR, DM = NH * DH;
  __shared__ float tile[32][33];
  const int r0 = blockIdx.y * 32, c0 = blockIdx.x * 32;
  const int tx = threadIdx.x % 32, ty = threadIdx.x / 32;
  const int c = c0 + tx, h = c / DH, d = c % DH;
  const bool live = d < DR;
  const int sc = h * DR + d;
  const float bq = live ? BQ[sc] : 0.f;
#pragma unroll
  for (int i = 0; i < 32; i += 8) {
    const int r = r0 + ty + i;
    float q = 0.f, k = 0.f, v = 0.f;
    if (live) {
      const float* row = QKVG + (long)r * 4 * N;
      q = row[sc] + bq;
      if (QF != nullptr) QF[(long)r * N + sc] = q;
      q *= scale;
      k = row[N + sc];
      v = row[2 * N + sc];
    }
    v = rna_tf32(v);
    QS[(long)r * DM + c] = rna_tf32(q);
    KR[(long)r * DM + c] = rna_tf32(k);
    tile[ty + i][tx] = v;
  }
  __syncthreads();
#pragma unroll
  for (int i = 0; i < 32; i += 8) VT[(long)(c0 + ty + i) * L + r0 + tx] = tile[tx][ty + i];
}

// OG [L, N] = sigmoid(g + gb) O  (O [L, DM] padded)
template <int NH, int DR, int DH>
__global__ void __launch_bounds__(256) gate_k(const float* __restrict__ O, const float* __restrict__ QKVG, float* __restrict__ OG, long total, float gb) {
  constexpr int N = NH * DR, DM = NH * DH;
  for (long i = (long)blockIdx.x * 256 + threadIdx.x; i < total; i += (long)gridDim.x * 256) {
    const long row = i / N;
    const int col = (int)(i % N), h = col / DR, d = col % DR;
    OG[i] = O[row * DM + h * DH + d] * sigf(QKVG[row * 4 * N + 3 * N + col] + gb);
  }
}

// backward, exact fp32 core (the attention gradients run on cuBLAS fp32 GEMMs between these rows: the TF32 core's dS / dP roundings put dq, dk
// at 2.4x the error of PyTorch's module, which is limited by its projection GEMMs only).
// DO [L, N] = d og sigmoid(g + gb), the g slot of the packed gradient DQKVG [L, 4N] = d og O sigmoid' (a thread per element)
template <int NH, int DR, int DH>
__global__ void __launch_bounds__(256) bwd_prep_k(const float* __restrict__ DOG, const float* __restrict__ O, const float* __restrict__ QKVG,
                                                  float* __restrict__ DO, float* __restrict__ DQKVG, long total, float gb) {
  constexpr int N = NH * DR, DM = NH * DH;
  for (long i = (long)blockIdx.x * 256 + threadIdx.x; i < total; i += (long)gridDim.x * 256) {
    const long row = i / N;
    const int col = (int)(i % N), h = col / DR, d = col % DR;
    const float dog = DOG[i], o = O[row * DM + h * DH + d];
    const float s = sigf(QKVG[row * 4 * N + 3 * N + col] + gb);
    DO[i] = dog * s;
    DQKVG[row * 4 * N + 3 * N + col] = dog * o * s * (1.f - s);
  }
}

// a warp per (head, query row): S [NH, L, L] = q k^T (natural, unscaled) and dP = dO v^T from the GEMMs, BIAS [NH, L, L] in log2 units in the core's key
// order. P = softmax over the keys (exact fp32: its own max and sum), D = sum_j P dP, dS = P (dP - D): P and dS are written, dS is dbias too.
template <int LMAX>
__global__ void __launch_bounds__(256) bwd_rows_k(const float* __restrict__ S, const float* __restrict__ DP, const float* __restrict__ BIAS,
                                                  float* __restrict__ P, float* __restrict__ DS, long rows, int L, float c) {
  constexpr int T = LMAX / 32;
  const int lane = threadIdx.x & 31;
  for (long w = (long)blockIdx.x * 8 + (threadIdx.x >> 5); w < rows; w += (long)gridDim.x * 8) {
    const float* s = S + w * L;
    const float* dp = DP + w * L;
    const float* b = BIAS + w * L;
    float v[T], d[T];
    float m = -INFINITY;
#pragma unroll
    for (int t = 0; t < T; ++t) {
      const int j = lane + 32 * t;
      if (j < L) {
        v[t] = s[j] * c + b[kinv(j)];
        d[t] = dp[j];
        m = fmaxf(m, v[t]);
      } else { v[t] = -INFINITY; d[t] = 0.f; }
    }
    for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    float l = 0.f;
#pragma unroll
    for (int t = 0; t < T; ++t) { v[t] = exp2f(v[t] - m); l += v[t]; }
    const float rl = 1.f / tdr::warp_sum(l);
    float dd = 0.f;
#pragma unroll
    for (int t = 0; t < T; ++t) { v[t] *= rl; dd += v[t] * d[t]; }
    dd = tdr::warp_sum(dd);
#pragma unroll
    for (int t = 0; t < T; ++t) {
      const int j = lane + 32 * t;
      if (j < L) { P[w * L + j] = v[t]; DS[w * L + j] = v[t] * (d[t] - dd); }
    }
  }
}

// dq, dk, dv [NH, L, DR] (the GEMMs' output) -> DQKVG [L, 4N] columns 0 .. 3N; dq, dk x qsc (1 / sqrt DR)
template <int NH, int DR>
__global__ void __launch_bounds__(256) post_k(const float* __restrict__ DQ, const float* __restrict__ DK, const float* __restrict__ DV,
                                              float* __restrict__ DQKVG, long total, int L, float qsc) {
  constexpr int N = NH * DR;
  for (long i = (long)blockIdx.x * 256 + threadIdx.x; i < total; i += (long)gridDim.x * 256) {
    const long row = i / N;
    const int col = (int)(i % N), h = col / DR, d = col % DR;
    const long src = ((long)h * L + row) * DR + d;
    float* o = DQKVG + row * 4 * N;
    o[col] = DQ[src] * qsc;
    o[N + col] = DK[src] * qsc;
    o[2 * N + col] = DV[src];
  }
}


// -------------------------------------------------------------------------------------------------- pair kernels, tensor cores
__device__ __forceinline__ void mma_tf32(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t tf32r(float x) { uint32_t r; asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(x)); return r; }
__device__ __forceinline__ float el4(const float4& v, int i) { return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w; }
// forward K slot s (< 8) of k-step kk: lane s % 4 of a row holds columns 16 m + 4 (s % 4) + {0..3} of chunk m = kk / 2 (its float4
// loads: 64 contiguous bytes per row across the four lanes), element 2 (kk % 2) + s / 4 -- the reduction order is free
__device__ __forceinline__ int fcol(int kk, int s) { return 16 * (kk / 2) + 4 * (s % 4) + 2 * (kk % 2) + s / 4; }

constexpr int WPB = 8;

// pair [R, DP] (R = L L) -> bias [NH, R] = LN(pair) Wf^T in the core's key order (perm), masked keys NEG; PST [R, 2] = (mean, rstd) if given.
// A warp takes 16 pair rows at a time (rows g, g + 8 per lane, all loads first), the LayerNorm statistics over the row's four lanes
// (two passes, exact), LN(pair) Wf^T on m16n8k8 TF32 (NT n-tiles of 8 heads; heads beyond NH carry zero weights).
template <int NH, int DP>
__global__ void __launch_bounds__(256, DP == 128 ? 2 : 1) pb_fwd_k(const float* __restrict__ Z, const float* __restrict__ WF, float* __restrict__ BIAS,
                                                                   float2* __restrict__ PST, long R, float eps, const bool* __restrict__ MASK,
                                                                   int L, float NEG) {
  constexpr int NT = (NH + 7) / 8, KS = DP / 8, MC = DP / 16;
  __shared__ uint4 bw[KS][NT][32];                       // Wf^T fragments per k-step, n-tile, lane: (b0, b1) rounded
  const int lane = threadIdx.x % 32, g = lane / 4, q = lane % 4;
  const long wid = (long)blockIdx.x * WPB + threadIdx.x / 32, nw = (long)gridDim.x * WPB;
  for (int i = threadIdx.x; i < KS * NT * 32; i += blockDim.x) {
    const int l = i % 32, t = (i / 32) % NT, kk = i / (32 * NT), h = 8 * t + l / 4;
    bw[kk][t][l] = make_uint4(tf32r(h < NH ? WF[h * DP + fcol(kk, l % 4)] : 0.f), tf32r(h < NH ? WF[h * DP + fcol(kk, l % 4 + 4)] : 0.f), 0u, 0u);
  }
  __syncthreads();
  for (long rb = wid * 16; rb < R; rb += nw * 16) {
    float4 u[2][MC];
#pragma unroll
    for (int hr = 0; hr < 2; ++hr)
#pragma unroll
      for (int m = 0; m < MC; ++m) u[hr][m] = ld4(Z + (rb + g + 8 * hr) * DP + 4 * (q + 4 * m));
    float mean[2], rs[2];
#pragma unroll
    for (int hr = 0; hr < 2; ++hr) {
      float s1 = 0.f;
#pragma unroll
      for (int m = 0; m < MC; ++m) s1 += sum4(u[hr][m]);
      s1 += __shfl_xor_sync(0xffffffffu, s1, 1); s1 += __shfl_xor_sync(0xffffffffu, s1, 2);
      mean[hr] = s1 * (1.f / DP);
      float s2 = 0.f;
#pragma unroll
      for (int m = 0; m < MC; ++m) {
        const float dx = u[hr][m].x - mean[hr], dy = u[hr][m].y - mean[hr], dz = u[hr][m].z - mean[hr], dw = u[hr][m].w - mean[hr];
        s2 += (dx * dx + dy * dy) + (dz * dz + dw * dw);
      }
      s2 += __shfl_xor_sync(0xffffffffu, s2, 1); s2 += __shfl_xor_sync(0xffffffffu, s2, 2);
      rs[hr] = rsqrtf(s2 * (1.f / DP) + eps);
    }
    float acc[NT][4];
#pragma unroll
    for (int t = 0; t < NT; ++t) acc[t][0] = acc[t][1] = acc[t][2] = acc[t][3] = 0.f;
#pragma unroll
    for (int kk = 0; kk < KS; ++kk) {
      const int m = kk / 2, e = 2 * (kk % 2);
      const float4 c0 = u[0][m], c1 = u[1][m];
      const uint32_t a0 = tf32r((el4(c0, e) - mean[0]) * rs[0]), a1 = tf32r((el4(c1, e) - mean[1]) * rs[1]),
                     a2 = tf32r((el4(c0, e + 1) - mean[0]) * rs[0]), a3 = tf32r((el4(c1, e + 1) - mean[1]) * rs[1]);
#pragma unroll
      for (int t = 0; t < NT; ++t) {
        const uint4 b = bw[kk][t][lane];
        mma_tf32(acc[t], a0, a1, a2, a3, b.x, b.y);
      }
    }
#pragma unroll
    for (int t = 0; t < NT; ++t)
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        const int h = 8 * t + 2 * q + e;
        if (h < NH) {
#pragma unroll
          for (int hr = 0; hr < 2; ++hr) {
            const long rr = rb + g + 8 * hr;
            const int j = (int)(rr % L);
            const int pj = kinv(j);
            const bool live = MASK == nullptr || MASK[j];
            BIAS[(long)h * R + (rr / L) * L + pj] = live ? acc[t][2 * hr + e] : NEG;
          }
        }
      }
    if (PST != nullptr && q == 0) { PST[rb + g] = make_float2(mean[0], rs[0]); PST[rb + g + 8] = make_float2(mean[1], rs[1]); }
  }
}

__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst), "l"(src) : "memory");
}
__device__ __forceinline__ void cp_async4(uint32_t dst, const void* src) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" :: "r"(dst), "l"(src) : "memory");
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
__device__ __forceinline__ void cp_async_wait1() { asm volatile("cp.async.wait_group 1;" ::: "memory"); }

template <int NH> struct PB32 {
  static constexpr int NK = (NH + 7) / 8;              // 8-head k-steps (heads beyond NH padded with zeros)
  static constexpr int NHP = 8 * (NK < 2 ? 2 : NK);    // at least two k-steps of the dbias^T operand
  static constexpr int NKK = NHP / 8;
  static constexpr int ZP = 136, DP = 132;
  static constexpr int ZS = 128 * ZP * 4, DS = NHP * DP * 4, BW = NKK * 16 * 32 * 16;
  static constexpr int SMEM = 2 * ZS + 2 * DS + BW;
};

// backward, d_pair 128: 128 pair rows per block item (16 per warp), streamed into shared memory by cp.async (double buffer). d LN(pair) = dbias^T Wf
// on mma (M = the warp's 16 rows, N = 16 channel tiles of 8, K = the heads), the LayerNorm backward with two row sums over the row's four lanes,
// d pair out; LN(pair) written back in place for dWf^T[16 w .., heads] += LN(pair)^T dbias^T over the item's rows. dWf leaves as one [NH, 128]
// partial per block. DB is the dbias in natural order; L % 128 == 0.
template <int NH>
__global__ void __launch_bounds__(256, 2) pb_bwd_tc_k(const float* __restrict__ DB, const float* __restrict__ Z, const float2* __restrict__ PST,
                                                      const float* __restrict__ WF, float* __restrict__ DZ, float* __restrict__ DWF, long R) {
  using C = PB32<NH>;
  constexpr int NKK = C::NKK, ZP = C::ZP, DP = C::DP, NHP = C::NHP;
  extern __shared__ __align__(16) uint8_t smem[];
  uint4 (*bw)[16][32] = reinterpret_cast<uint4 (*)[16][32]>(smem + 2 * C::ZS + 2 * C::DS);   // Wf fragments [k-step][channel tile][lane]
  const uint32_t s_base = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  const int lane = threadIdx.x % 32, w = threadIdx.x / 32, g = lane / 4, q = lane % 4;
  auto prefetch = [&](long b0, int b) {
    for (int i = threadIdx.x; i < 128 * 32; i += blockDim.x) {
      const int row = i / 32, c = i % 32;
      cp_async16(s_base + b * C::ZS + (row * ZP + 4 * c) * 4, Z + (b0 + row) * 128 + 4 * c);
    }
    for (int i = threadIdx.x; i < NH * 128; i += blockDim.x) {
      const int h = i / 128, j = i % 128;
      cp_async4(s_base + 2 * C::ZS + b * C::DS + (h * DP + j) * 4, DB + (long)h * R + b0 + j);
    }
  };
  for (int i = threadIdx.x; i < NKK * 16 * 32; i += blockDim.x) {    // B = Wf [heads (k) x channels (n)]: b0 (head 8 ks + q, channel 8 j + g)
    const int l = i % 32, j = (i / 32) % 16, ks = i / 512, k = 8 * ks + l % 4, c = 8 * j + l / 4;
    const float w0 = k < NH ? WF[k * 128 + c] : 0.f, w1 = k + 4 < NH ? WF[(k + 4) * 128 + c] : 0.f;
    bw[ks][j][l] = make_uint4(tf32r(w0), tf32r(w1), 0u, 0u);
  }
  for (int i = threadIdx.x; i < 2 * (NHP - NH) * DP; i += blockDim.x) {   // the padding heads of both dbias buffers weigh 0
    const int b = i / ((NHP - NH) * DP), k = i % ((NHP - NH) * DP);
    reinterpret_cast<float*>(smem + 2 * C::ZS + b * C::DS)[NH * DP + k] = 0.f;
  }
  float dacc[NKK][4];
#pragma unroll
  for (int t = 0; t < NKK; ++t) dacc[t][0] = dacc[t][1] = dacc[t][2] = dacc[t][3] = 0.f;
  const long step = (long)gridDim.x * 128;
  if ((long)blockIdx.x * 128 < R) prefetch((long)blockIdx.x * 128, 0);
  cp_async_commit();
  int b = 0;
  for (long b0 = (long)blockIdx.x * 128; b0 < R; b0 += step, b ^= 1) {
    if (b0 + step < R) prefetch(b0 + step, b ^ 1);
    cp_async_commit();
    const int wr = 16 * w;
    const long rb = b0 + wr;
    const float2 st0 = PST[rb + g], st1 = PST[rb + g + 8];
    cp_async_wait1();                                  // this item's group (the next one may still be in flight)
    __syncthreads();
    float* zs = reinterpret_cast<float*>(smem + b * C::ZS);
    const float* ds = reinterpret_cast<const float*>(smem + 2 * C::ZS + b * C::DS);
    // A = dbias^T [16 rows x heads]: a0 (row g, head 8 ks + q), a1 (row g + 8, ..), a2 (row g, head 8 ks + q + 4), a3
    uint32_t ah[NKK][4];
#pragma unroll
    for (int ks = 0; ks < NKK; ++ks) {
      const float* d0 = ds + (8 * ks + q) * DP + wr + g;
      ah[ks][0] = tf32r(ldsv(d0)); ah[ks][1] = tf32r(ldsv(d0 + 8)); ah[ks][2] = tf32r(ldsv(d0 + 4 * DP)); ah[ks][3] = tf32r(ldsv(d0 + 4 * DP + 8));
    }
    const float* z0 = zs + (wr + g) * ZP + 2 * q;
    const float* z1 = z0 + 8 * ZP;
    // pass 1: d LN for every channel tile (kept), the row sums of d LN and of d LN (pair - mean)
    float dk[16][4];
    float sd[2] = {0.f, 0.f}, sdv[2] = {0.f, 0.f};
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      float d[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int ks = 0; ks < NKK; ++ks) {
        const uint4 bb = bw[ks][j][lane];
        mma_tf32(d, ah[ks][0], ah[ks][1], ah[ks][2], ah[ks][3], bb.x, bb.y);
      }
      dk[j][0] = d[0]; dk[j][1] = d[1]; dk[j][2] = d[2]; dk[j][3] = d[3];
      const float2 x0 = *reinterpret_cast<const float2*>(z0 + 8 * j), x1 = *reinterpret_cast<const float2*>(z1 + 8 * j);
      sd[0] += d[0] + d[1]; sdv[0] = fmaf(d[0], x0.x - st0.x, fmaf(d[1], x0.y - st0.x, sdv[0]));
      sd[1] += d[2] + d[3]; sdv[1] = fmaf(d[2], x1.x - st1.x, fmaf(d[3], x1.y - st1.x, sdv[1]));
    }
    // d pair = rstd (d LN - mean(d LN) - LN mean(d LN LN)) = rstd d LN - c1 (pair - mean) + c0
    float c1[2], c0[2];
    const float2 st[2] = {st0, st1};
#pragma unroll
    for (int k = 0; k < 2; ++k) {
      sd[k] += __shfl_xor_sync(0xffffffffu, sd[k], 1); sd[k] += __shfl_xor_sync(0xffffffffu, sd[k], 2);
      sdv[k] += __shfl_xor_sync(0xffffffffu, sdv[k], 1); sdv[k] += __shfl_xor_sync(0xffffffffu, sdv[k], 2);
      c1[k] = st[k].y * st[k].y * st[k].y * sdv[k] * (1.f / 128);
      c0[k] = -st[k].y * sd[k] * (1.f / 128);
    }
    // pass 2: d pair out; LN(pair) back in place (this warp's rows only: the dWf product reads them after the barrier)
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      float2 x0 = *reinterpret_cast<const float2*>(z0 + 8 * j), x1 = *reinterpret_cast<const float2*>(z1 + 8 * j);
      x0.x -= st0.x; x0.y -= st0.x; x1.x -= st1.x; x1.y -= st1.x;
      *reinterpret_cast<float2*>(DZ + (rb + g) * 128 + 8 * j + 2 * q) =
          make_float2(fmaf(st0.y, dk[j][0], fmaf(-c1[0], x0.x, c0[0])), fmaf(st0.y, dk[j][1], fmaf(-c1[0], x0.y, c0[0])));
      *reinterpret_cast<float2*>(DZ + (rb + g + 8) * 128 + 8 * j + 2 * q) =
          make_float2(fmaf(st1.y, dk[j][2], fmaf(-c1[1], x1.x, c0[1])), fmaf(st1.y, dk[j][3], fmaf(-c1[1], x1.y, c0[1])));
      *reinterpret_cast<float2*>(zs + (wr + g) * ZP + 8 * j + 2 * q) = make_float2(x0.x * st0.y, x0.y * st0.y);
      *reinterpret_cast<float2*>(zs + (wr + g + 8) * ZP + 8 * j + 2 * q) = make_float2(x1.x * st1.y, x1.y * st1.y);
    }
    __syncthreads();                                   // LN of the item's rows
    // dWf^T [16 w + m, head 8 t + n] over the item's rows: A = LN^T (m channel, k row), B = dbias^T (k row, n head)
#pragma unroll 4
    for (int kk = 0; kk < 16; ++kk) {
      const int k0 = 8 * kk;
      const float* a0p = zs + (k0 + q) * ZP + 16 * w + g;
      const uint32_t x0 = tf32r(a0p[0]), x1 = tf32r(a0p[8]), x2 = tf32r(a0p[4 * ZP]), x3 = tf32r(a0p[4 * ZP + 8]);
#pragma unroll
      for (int t = 0; t < NKK; ++t) {
        const float* bp = ds + (8 * t + g) * DP + k0 + q;
        mma_tf32(dacc[t], x0, x1, x2, x3, tf32r(bp[0]), tf32r(bp[4]));
      }
    }
    __syncthreads();                                   // before this buffer is refilled
  }
  // dacc[t]: d0, d1 = channel 16 w + g, heads 8 t + 2 q + {0, 1}; d2, d3 = channel 16 w + g + 8
#pragma unroll
  for (int t = 0; t < NKK; ++t)
#pragma unroll
    for (int e = 0; e < 2; ++e) {
      const int h = 8 * t + 2 * q + e;
      if (h < NH) {
        DWF[((long)blockIdx.x * NH + h) * 128 + 16 * w + g] = dacc[t][e];
        DWF[((long)blockIdx.x * NH + h) * 128 + 16 * w + g + 8] = dacc[t][2 + e];
      }
    }
}

// ------------------------------------------------------------------------------------------ pair backward, FMA pipe (d_pair 256, 8 heads)
constexpr int BT = 256, RWPB = BT / 32, PF = 4;
template <int DPL>
__device__ __forceinline__ void load_row(const float* __restrict__ z, float (&x)[DPL]) {
#pragma unroll
  for (int i = 0; i < DPL / 4; ++i) {
    const float4 a = __ldg(reinterpret_cast<const float4*>(z) + i);
    x[4 * i] = a.x; x[4 * i + 1] = a.y; x[4 * i + 2] = a.z; x[4 * i + 3] = a.w;
  }
}
template <int DPL>
__device__ __forceinline__ void store_row(float* __restrict__ z, const float (&x)[DPL]) {
#pragma unroll
  for (int i = 0; i < DPL / 4; ++i) reinterpret_cast<float4*>(z)[i] = make_float4(x[4 * i], x[4 * i + 1], x[4 * i + 2], x[4 * i + 3]);
}
template <int DP, int DPL>
__device__ __forceinline__ void normalize(float (&x)[DPL], float eps, float& rstd) {
  float s = 0.f;
#pragma unroll
  for (int i = 0; i < DPL; ++i) s += x[i];
  const float mean = tdr::warp_sum(s) * (1.f / DP);
  float v = 0.f;
#pragma unroll
  for (int i = 0; i < DPL; ++i) { x[i] -= mean; v += x[i] * x[i]; }
  rstd = rsqrtf(tdr::warp_sum(v) * (1.f / DP) + eps);
#pragma unroll
  for (int i = 0; i < DPL; ++i) x[i] *= rstd;
}

template <int DP, int NH>
__global__ void __launch_bounds__(BT) pb_bwd_rows_k(const float* __restrict__ Z, const float* __restrict__ DB, const float* __restrict__ W,
                                                    float* __restrict__ DZ, float* __restrict__ ACC, long R, float eps) {
  constexpr int DPL = DP / 32;
  __shared__ float red[RWPB][DP];
  const int lane = threadIdx.x & 31, wp = threadIdx.x >> 5;
  float w[NH][DPL], part[NH][DPL];
#pragma unroll
  for (int h = 0; h < NH; ++h)
#pragma unroll
    for (int i = 0; i < DPL; ++i) { w[h][i] = W[h * DP + lane * DPL + i]; part[h][i] = 0.f; }
  const long warp = ((long)blockIdx.x * BT + threadIdx.x) >> 5, nwarps = (long)gridDim.x * RWPB;
  for (long base = warp * 32; base < R; base += nwarps * 32) {
    float dbr[NH];
    const long mine = base + lane;
    const long src = mine;
#pragma unroll
    for (int h = 0; h < NH; ++h) dbr[h] = mine < R ? __ldg(DB + (long)h * R + src) : 0.f;
    const int n = (int)min(32L, R - base);
    for (int rb = 0; rb < n; rb += PF) {
      float xb[PF][DPL];
#pragma unroll
      for (int q = 0; q < PF; ++q) load_row<DPL>(Z + min(base + rb + q, R - 1) * DP + lane * DPL, xb[q]);
#pragma unroll
      for (int q = 0; q < PF; ++q) {
        const int r = rb + q;
        if (r >= n) break;
        float x[DPL], d[NH], g[DPL];
#pragma unroll
        for (int i = 0; i < DPL; ++i) x[i] = xb[q][i];
        float rstd;
        normalize<DP, DPL>(x, eps, rstd);
#pragma unroll
        for (int h = 0; h < NH; ++h) d[h] = __shfl_sync(0xffffffffu, dbr[h], r);
        float mg = 0.f, mgx = 0.f;
#pragma unroll
        for (int i = 0; i < DPL; ++i) {
          float s = 0.f;
#pragma unroll
          for (int h = 0; h < NH; ++h) { s += w[h][i] * d[h]; part[h][i] += d[h] * x[i]; }
          g[i] = s;
          mg += s;
          mgx += s * x[i];
        }
        mg = tdr::warp_sum(mg) * (1.f / DP);
        mgx = tdr::warp_sum(mgx) * (1.f / DP);
#pragma unroll
        for (int i = 0; i < DPL; ++i) g[i] = rstd * (g[i] - mg - x[i] * mgx);
        store_row<DPL>(DZ + (base + r) * DP + lane * DPL, g);
      }
    }
  }
  for (int h = 0; h < NH; ++h) {
#pragma unroll
    for (int i = 0; i < DPL; ++i) red[wp][lane * DPL + i] = part[h][i];
    __syncthreads();
    for (int c = threadIdx.x; c < DP; c += BT) {
      float v = 0.f;
#pragma unroll
      for (int q = 0; q < RWPB; ++q) v += red[q][c];
      ACC[((long)blockIdx.x * NH + h) * DP + c] = v;
    }
    __syncthreads();
  }
}

// ------------------------------------------------------------------------------------------------------------------------- host
int nsm() { static int n = at::cuda::getCurrentDeviceProperties()->multiProcessorCount; return n; }
cudaStream_t S() { return at::cuda::getCurrentCUDAStream(); }
void f32c(const at::Tensor& t, const char* n) { TORCH_CHECK(t.is_cuda() && t.scalar_type() == at::kFloat && t.is_contiguous(), n, ": contiguous fp32 CUDA tensor"); }

// (heads, real width) -> padded width; the layouts the attention core is built for
#define LAYOUT_SWITCH(NHV, DRV, ...)                                                                                      \
  do {                                                                                                                    \
    if ((NHV) == 8 && (DRV) == 48) { constexpr int NHK = 8, DRK = 48, DHK = 48; __VA_ARGS__; }                            \
    else if ((NHV) == 12 && (DRV) == 32) { constexpr int NHK = 12, DRK = 32, DHK = 32; __VA_ARGS__; }                     \
    else if ((NHV) == 16 && (DRV) == 24) { constexpr int NHK = 16, DRK = 24, DHK = 32; __VA_ARGS__; }                     \
    else if ((NHV) == 16 && (DRV) == 32) { constexpr int NHK = 16, DRK = 32, DHK = 32; __VA_ARGS__; }                     \
    else if ((NHV) == 24 && (DRV) == 16) { constexpr int NHK = 24, DRK = 16, DHK = 32; __VA_ARGS__; }                     \
    else TORCH_CHECK(false, "apb_tf32: unsupported head layout ", (long)(NHV), " x ", (long)(DRV));                       \
  } while (0)

std::vector<at::Tensor> apb_prep(at::Tensor qkvg, at::Tensor bq, int64_t nh, int64_t dr, double scale, bool want_qf) {
  f32c(qkvg, "qkvg"); f32c(bq, "bq");
  const int64_t L = qkvg.size(0);
  const at::cuda::CUDAGuard gd(qkvg.device());
  std::vector<at::Tensor> out;
  LAYOUT_SWITCH(nh, dr, {
    constexpr int DM = NHK * DHK;
    TORCH_CHECK(qkvg.size(1) == 4 * NHK * DRK && bq.numel() == NHK * DRK && L % 32 == 0 && DM % 32 == 0, "apb_prep shapes");
    auto qs = at::empty({L, DM}, qkvg.options()), kr = at::empty_like(qs), vt = at::empty({DM, L}, qkvg.options());
    auto qf = want_qf ? at::empty({L, NHK * DRK}, qkvg.options()) : at::empty({0}, qkvg.options());
    prep_k<NHK, DRK, DHK><<<dim3(DM / 32, (unsigned)(L / 32)), 256, 0, S()>>>(qkvg.data_ptr<float>(), bq.data_ptr<float>(), qs.data_ptr<float>(),
        kr.data_ptr<float>(), want_qf ? qf.data_ptr<float>() : nullptr, vt.data_ptr<float>(), (int)L, (float)scale);
    out = {qs, kr, qf, vt};
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

at::Tensor apb_gate(at::Tensor o, at::Tensor qkvg, int64_t nh, int64_t dr, double gb) {
  f32c(o, "o"); f32c(qkvg, "qkvg");
  const int64_t L = qkvg.size(0);
  const at::cuda::CUDAGuard gd(o.device());
  at::Tensor og;
  LAYOUT_SWITCH(nh, dr, {
    constexpr int N = NHK * DRK;
    TORCH_CHECK(qkvg.size(1) == 4 * N && o.size(0) == L && o.size(1) == NHK * DHK, "apb_gate shapes");
    og = at::empty({L, N}, o.options());
    const long total = (long)L * N;
    gate_k<NHK, DRK, DHK><<<(unsigned)std::min<long>((total + 255) / 256, 8L * nsm()), 256, 0, S()>>>(o.data_ptr<float>(), qkvg.data_ptr<float>(), og.data_ptr<float>(), total, (float)gb);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return og;
}

// (dO [L, N], dqkvg [L, 4N] with the g slot written)
std::vector<at::Tensor> apb_bwd_prep(at::Tensor dog, at::Tensor o, at::Tensor qkvg, int64_t nh, int64_t dr, double gb) {
  f32c(dog, "dog"); f32c(o, "o"); f32c(qkvg, "qkvg");
  const int64_t L = qkvg.size(0);
  const at::cuda::CUDAGuard gd(o.device());
  std::vector<at::Tensor> out;
  LAYOUT_SWITCH(nh, dr, {
    constexpr int N = NHK * DRK, DM = NHK * DHK;
    TORCH_CHECK(dog.size(0) == L && dog.size(1) == N && o.size(1) == DM, "apb_bwd_prep shapes");
    auto dO = at::empty({L, N}, o.options());
    auto dqkvg = at::empty({L, 4 * N}, o.options());
    const long total = (long)L * N;
    bwd_prep_k<NHK, DRK, DHK><<<(unsigned)std::min<long>((total + 255) / 256, 8L * nsm()), 256, 0, S()>>>(dog.data_ptr<float>(), o.data_ptr<float>(), qkvg.data_ptr<float>(),
        dO.data_ptr<float>(), dqkvg.data_ptr<float>(), total, (float)gb);
    out = {dO, dqkvg};
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

// s, dp [nh, L, L] (q k^T unscaled, dO v^T), bias_p [nh, L, L] (log2 units, key-permuted), c = log2 e / sqrt(dr) -> (P, dS) natural key order
std::vector<at::Tensor> apb_bwd_rows(at::Tensor s, at::Tensor dp, at::Tensor bias_p, double c) {
  f32c(s, "s"); f32c(dp, "dp"); f32c(bias_p, "bias");
  const int64_t L = s.size(2), rows = s.size(0) * s.size(1);
  TORCH_CHECK(s.dim() == 3 && s.size(1) == L && dp.sizes() == s.sizes() && bias_p.sizes() == s.sizes() && L % 8 == 0 && L <= 1536, "apb_bwd_rows shapes (L <= 1536)");
  const at::cuda::CUDAGuard gd(s.device());
  auto p = at::empty_like(s), ds = at::empty_like(s);
  const unsigned g = (unsigned)std::min<int64_t>((rows + 7) / 8, 16L * nsm());
  auto launch = [&](auto kern) {
    kern<<<g, 256, 0, S()>>>(s.data_ptr<float>(), dp.data_ptr<float>(), bias_p.data_ptr<float>(), p.data_ptr<float>(), ds.data_ptr<float>(), rows, (int)L, (float)c);
  };
  if (L <= 512) launch(bwd_rows_k<512>); else if (L <= 1024) launch(bwd_rows_k<1024>); else launch(bwd_rows_k<1536>);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {p, ds};
}

void apb_post(at::Tensor dq, at::Tensor dk, at::Tensor dv, at::Tensor dqkvg, int64_t nh, int64_t dr) {
  f32c(dq, "dq"); f32c(dk, "dk"); f32c(dv, "dv"); f32c(dqkvg, "dqkvg");
  const int64_t L = dqkvg.size(0);
  const at::cuda::CUDAGuard gd(dq.device());
  LAYOUT_SWITCH(nh, dr, {
    constexpr int N = NHK * DRK;
    TORCH_CHECK(dqkvg.size(1) == 4 * N && dq.numel() == (int64_t)NHK * L * DRK, "apb_post shapes");
    const long total = (long)L * N;
    post_k<NHK, DRK><<<(unsigned)std::min<long>((total + 255) / 256, 8L * nsm()), 256, 0, S()>>>(dq.data_ptr<float>(), dk.data_ptr<float>(), dv.data_ptr<float>(),
        dqkvg.data_ptr<float>(), total, (int)L, 1.f / sqrtf((float)DRK));
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int NH, int DP>
std::vector<at::Tensor> pb_fwd_launch(at::Tensor z, at::Tensor wf, c10::optional<at::Tensor> mask, int64_t L, double eps, double neg, bool want_pst) {
  const int64_t R = z.size(0);
  const bool* mp = nullptr;
  if (mask.has_value() && mask->defined()) {
    TORCH_CHECK(mask->scalar_type() == at::kBool && mask->is_contiguous() && mask->numel() == L, "pb_fwd: bool mask [L]");
    mp = mask->data_ptr<bool>();
  }
  auto bias = at::empty({NH, L, L}, z.options());
  auto pst = want_pst ? at::empty({R, 2}, z.options()) : at::empty({0}, z.options());
  auto k = pb_fwd_k<NH, DP>;
  int per = 1;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, k, 256, 0);
  const unsigned g = (unsigned)std::max<int64_t>(1, std::min<int64_t>((R / 16 + WPB - 1) / WPB, (int64_t)std::max(per, 1) * nsm()));
  k<<<g, 256, 0, S()>>>(z.data_ptr<float>(), wf.data_ptr<float>(), bias.data_ptr<float>(), want_pst ? reinterpret_cast<float2*>(pst.data_ptr<float>()) : nullptr,
                        R, (float)eps, mp, (int)L, (float)neg);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {bias, pst};
}

// pair [L L, dp] fp32, wf [nh, dp] (LayerNorm weight and head projection folded, log2 units) -> (bias [nh, L, L] key-permuted, stats [L L, 2])
std::vector<at::Tensor> pb_fwd(at::Tensor z, at::Tensor wf, c10::optional<at::Tensor> mask, int64_t L, double eps, double neg, bool want_pst) {
  f32c(z, "pair"); f32c(wf, "Wf");
  const int64_t nh = wf.size(0), dp = wf.size(1);
  TORCH_CHECK(z.dim() == 2 && z.size(1) == dp && z.size(0) == L * L && (L * L) % 16 == 0 && L % 8 == 0, "pb_fwd: pair [L L, dp]");
  const at::cuda::CUDAGuard gd(z.device());
  if (dp == 128) {
    if (nh == 8) return pb_fwd_launch<8, 128>(z, wf, mask, L, eps, neg, want_pst);
    if (nh == 12) return pb_fwd_launch<12, 128>(z, wf, mask, L, eps, neg, want_pst);
    if (nh == 16) return pb_fwd_launch<16, 128>(z, wf, mask, L, eps, neg, want_pst);
    if (nh == 24) return pb_fwd_launch<24, 128>(z, wf, mask, L, eps, neg, want_pst);
  } else if (dp == 256 && nh == 8) {
    return pb_fwd_launch<8, 256>(z, wf, mask, L, eps, neg, want_pst);
  }
  TORCH_CHECK(false, "pb_fwd: ", (long)nh, " heads x d_pair ", (long)dp, " not built");
  return {};
}

template <int NH>
std::vector<at::Tensor> pb_bwd_tc_launch(at::Tensor db, at::Tensor z, at::Tensor pst, at::Tensor wf) {
  using C = PB32<NH>;
  const int64_t R = z.size(0);
  auto k = pb_bwd_tc_k<NH>;
  cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)C::SMEM);
  int per = 1;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, k, 256, C::SMEM);
  const unsigned g = (unsigned)std::max<int64_t>(1, std::min<int64_t>({R / 128, (int64_t)std::max(per, 1) * nsm(), (int64_t)8 * nsm()}));
  auto dz = at::empty_like(z);
  auto dwf = at::empty({(int64_t)g, NH, 128}, z.options());
  k<<<g, 256, C::SMEM, S()>>>(db.data_ptr<float>(), z.data_ptr<float>(), reinterpret_cast<const float2*>(pst.data_ptr<float>()), wf.data_ptr<float>(), dz.data_ptr<float>(),
                              dwf.data_ptr<float>(), R);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {dz, dwf};
}

// db [nh, L, L] natural order, pair [L L, 128], stats from pb_fwd, wf [nh, 128] natural units -> (dpair, dWf partials [blocks, nh, 128])
std::vector<at::Tensor> pb_bwd_tc(at::Tensor db, at::Tensor z, at::Tensor pst, at::Tensor wf) {
  f32c(db, "dbias"); f32c(z, "pair"); f32c(pst, "pst"); f32c(wf, "Wf");
  const int64_t nh = wf.size(0), R = z.size(0);
  TORCH_CHECK(z.dim() == 2 && z.size(1) == 128 && wf.size(1) == 128 && R % 128 == 0 && db.numel() == nh * R && pst.numel() == 2 * R, "pb_bwd_tc shapes");
  TORCH_CHECK(db.size(1) % 128 == 0, "pb_bwd_tc: L a multiple of 128");
  const at::cuda::CUDAGuard gd(z.device());
  if (nh == 8) return pb_bwd_tc_launch<8>(db, z, pst, wf);
  if (nh == 12) return pb_bwd_tc_launch<12>(db, z, pst, wf);
  if (nh == 16) return pb_bwd_tc_launch<16>(db, z, pst, wf);
  if (nh == 24) return pb_bwd_tc_launch<24>(db, z, pst, wf);
  TORCH_CHECK(false, "pb_bwd_tc: heads");
  return {};
}

std::vector<at::Tensor> pb_bwd_rows(at::Tensor db, at::Tensor z, at::Tensor wf, double eps) {
  f32c(db, "dbias"); f32c(z, "pair"); f32c(wf, "Wf");
  const int64_t nh = wf.size(0), dp = wf.size(1), R = z.size(0);
  TORCH_CHECK(nh == 8 && dp == 256 && z.size(1) == 256 && db.numel() == nh * R && R % 32 == 0, "pb_bwd_rows: 8 heads, d_pair 256");
  const at::cuda::CUDAGuard gd(z.device());
  auto k = pb_bwd_rows_k<256, 8>;
  const unsigned g = (unsigned)std::min<int64_t>((R / 32 + RWPB - 1) / RWPB, 2L * nsm());
  auto dz = at::empty_like(z);
  auto acc = at::empty({(int64_t)g, 8, 256}, z.options());
  k<<<g, BT, 0, S()>>>(z.data_ptr<float>(), db.data_ptr<float>(), wf.data_ptr<float>(), dz.data_ptr<float>(), acc.data_ptr<float>(), R, (float)eps);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {dz, acc};
}
}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("apb_prep", &apb_prep);
  m.def("apb_gate", &apb_gate);
  m.def("apb_bwd_prep", &apb_bwd_prep);
  m.def("apb_post", &apb_post);
  m.def("apb_bwd_rows", &apb_bwd_rows);
  m.def("pb_fwd", &pb_fwd);
  m.def("pb_bwd_tc", &pb_bwd_tc);
  m.def("pb_bwd_rows", &pb_bwd_rows);
}
