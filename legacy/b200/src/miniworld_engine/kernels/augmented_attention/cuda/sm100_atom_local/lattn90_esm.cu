// lattn90_esm.cu — ESMFold2's windowed atom attention (inference) on sm_90a with mma.sync, bf16 and TF32.
//
// ESMFold2 gives every 32-query window a 192-key subset slid the way AlphaFold 3 slides its 128 (``AtomCrossAtt.compute_features``:
// [32 w - 80, 32 w + 112), a window short of atom 0 starts at 0, one past the last real atom ends at it), and lets query i see key j only
// when |i - j| <= 64, by rank among the real atoms -- which is the position, the real atoms coming first.
//
//   S = q k^T / sqrt(32) + bias[h][w][i][j - start_w]      O = softmax(S) v        (q, k RMS-normalised and rotated: qknorm_local.cu)
//
// For a real query the band [i - 64, i + 64] cut to the real atoms always lies inside its window's subset, so the kernels walk positions,
// not subset slots: the pair bias is re-indexed by position once per call (it is shared by the samples), the attention never sees a slot.
//   h_pair_bias_esm   bias2[h][w][i][c] = LN(z[w, i, p - start_w]) gamma . Wb[h] log2(e) at the position p = 32 w - 64 + c (c < 160), -inf
//                     where p is not a valid key, |p - (32 w + i)| > 64 or p - start_w falls outside the subset. start_w from the real atom
//                     count on the device (no host read: the call can be captured).
//   h_attn_esm        CTA = 64 queries (2 windows) x 1 head x a chunk of samples; K / V = the 192 positions [q0 - 64, q0 + 128); warp = 16
//                     queries over the 144 positions [r - 64, r + 80) of its rows (18 key tiles; the band edges are -inf in the bias).
//   h_attn_esm_tf32   the same on fp32 activations (TF32 products, m16n8k8, P rounded to nearest tf32), one stage.
//   h_attn_esm also writes the row LSE (log2 units) when given a pointer (training); the backward is
//   h_attn_esm_dq    same geometry: dQ (bf16 into the caller's [dq | dk | dv] columns) and the pair-bias gradient partial per sample chunk
//   h_attn_esm_dkv   CTA = 64 keys x 1 head x a chunk of samples; warp = 16 keys over the 144 queries [k - 64, k + 80): dK, dV
//   h_pair_bias_esm_bwd  dbias2 -> dz (the trunked pair's slots), per-block LayerNorm / projection gradient partials
// The TF32 kernel has no backward: fp32 ESMFold2 training runs the composed PyTorch path.
// SPDX-License-Identifier: Apache-2.0
#include <cuda_bf16.h>
#include <stdint.h>
#define DEVI __device__ __forceinline__

typedef __nv_bfloat16 bf16;

namespace {
constexpr int KEYS = 192, HALF = 64, SPAN = 160;   // subset, rank window, positions a window can see (32 + 2 x 64)
constexpr int WSPAN = 144;                         // positions a 16-query warp can see (16 + 2 x 64)
constexpr int BS = 168;    // fp32 row stride of the bias tile (160 + 8: conflict-free float2 reads)
constexpr int ST = 40;     // bf16 row stride of the staged tiles (80 B: conflict-free ldmatrix)
constexpr int ST32 = 36;   // fp32 row stride of the TF32 kernel's tiles
constexpr float SCALE = 0.17677669529663687f;
constexpr float LOG2E = 1.4426950408889634f;
constexpr float C1 = SCALE * LOG2E;

DEVI void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
DEVI uint32_t su(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
DEVI void cp16(uint32_t d, const void* s, bool ok) {
  const int n = ok ? 16 : 0;
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(d), "l"(s), "r"(n) : "memory");
}
DEVI void cp16f(uint32_t d, const void* s) { asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(d), "l"(s) : "memory"); }
DEVI void cp_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
template <int N> DEVI void cp_wait() { asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory"); }
DEVI void ldsm(uint32_t* r, uint32_t a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
DEVI void ldsmt(uint32_t* r, uint32_t a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
DEVI void mma(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
DEVI void mma_tf32(float* d, const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
DEVI uint32_t pk(float lo, float hi) {
  uint32_t r;
  asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(r) : "f"(lo), "f"(hi));
  return r;
}
DEVI uint32_t tf(float x) { uint32_t r; asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(x)); return r; }
DEVI float ex2(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
DEVI float quad_max(float v) {
  v = fmaxf(v, __shfl_xor_sync(~0u, v, 1));
  return fmaxf(v, __shfl_xor_sync(~0u, v, 2));
}
DEVI float quad_sum(float v) {
  v += __shfl_xor_sync(~0u, v, 1);
  return v + __shfl_xor_sync(~0u, v, 2);
}
// rows [r0, r0 + n) of head h of x[a] (x: [A, N, 128]) into shared rows of stride `st` elements; rows outside [0, N) are zeros
template <typename T>
DEVI void load_rows(uint32_t dst, const T* x, int a, int h, int r0, int n, int N, int st, int tid) {
  constexpr int PER = 16 / sizeof(T), CH = 32 / PER;   // elements per 16-byte chunk, chunks per head row
  for (int i = tid; i < n * CH; i += 128) {
    const int r = i / CH, c = i % CH, gr = r0 + r;
    const bool ok = gr >= 0 && gr < N;
    cp16(dst + (r * st + c * PER) * sizeof(T), x + ((size_t)a * N + (ok ? gr : 0)) * 128 + h * 32 + c * PER, ok);
  }
}
// the bias of the CTA's two windows [64 rows][160] -> shared rows of stride BS
DEVI void load_bias(float* sb, const float* bias, int h, int nwin, int w0, int tid) {
  for (int i = tid; i < 64 * (SPAN / 4); i += 128) {
    const int r = i / (SPAN / 4), c = i % (SPAN / 4);
    cp16f(su(sb + r * BS + c * 4), bias + ((size_t)(h * nwin + w0) * 32 + r) * SPAN + c * 4);
  }
}
// vec[a, h, r0 .. r0 + n) fp32 (LSE / D) into shared floats by cp.async (16-byte chunks; r0 and n multiples of 4); out-of-range -> 0
DEVI void load_vec(float* dst, const float* v, int a, int h, int r0, int n, int N, int tid) {
  for (int i = tid; i < n / 4; i += 128) {
    const int gr = r0 + 4 * i;
    const bool ok = gr >= 0 && gr < N;
    cp16(su(dst + 4 * i), v + ((size_t)a * 4 + h) * N + (ok ? gr : 0), ok);
  }
}
// one warp's 16 x 32 bf16 result tile (times scale) -> global rows of ld elements through the warp's private shared rows (stride 80 B)
DEVI void store16x32(uint8_t* stg, const float (*d)[4], bf16* row0, size_t ld, int lane, float scale = 1.f) {
  const int g = lane >> 2, t = lane & 3;
#pragma unroll
  for (int n = 0; n < 4; ++n) {
    *reinterpret_cast<uint32_t*>(stg + g * 80 + n * 16 + t * 4) = pk(d[n][0] * scale, d[n][1] * scale);
    *reinterpret_cast<uint32_t*>(stg + (g + 8) * 80 + n * 16 + t * 4) = pk(d[n][2] * scale, d[n][3] * scale);
  }
  __syncwarp();
#pragma unroll
  for (int c = lane; c < 64; c += 32) {
    const int row = c >> 2, part = c & 3;
    *reinterpret_cast<uint4*>(row0 + (size_t)row * ld + part * 8) = *reinterpret_cast<const uint4*>(stg + row * 80 + part * 16);
  }
  __syncwarp();
}
}  // namespace

// ------------------------------------------------------------------------------------------------------------------ pair bias
// z: the trunked pair [nwin, 32, 192, 16] bf16 (zbf16 = 1) or fp32; mask: the valid keys [N] (uint8) or null; nreal: the real atom count
// (a device int: the AF3 slide reads it); a thread per (w, i, c) entry of bias2 [4][nwin][32][160].
extern "C" __global__ __launch_bounds__(256) void h_pair_bias_esm(const void* __restrict__ z, const float* __restrict__ gamma,
                                                                   const float* __restrict__ wb, const uint8_t* __restrict__ mask,
                                                                   const int* __restrict__ nreal, float* __restrict__ out, int N, float eps,
                                                                   int zbf16) {
  pdl_wait();
  const int nwin = N / 32, nr = *nreal;
  const size_t count = (size_t)nwin * 32 * SPAN;
  for (size_t e = (size_t)blockIdx.x * blockDim.x + threadIdx.x; e < count; e += (size_t)gridDim.x * blockDim.x) {
    const int c = (int)(e % SPAN), i = (int)((e / SPAN) % 32), w = (int)(e / (32 * SPAN));
    const int p = 32 * w - HALF + c, qp = 32 * w + i;
    const int first = 32 * w - (KEYS - 32) / 2;
    const int start = first < 0 ? 0 : min(first, nr - KEYS);           // AF3: may begin before atom 0 below 192 real atoms (padding there)
    const int slot = p - start;
    const bool ok = p >= 0 && p < N && p < nr && (mask == nullptr || mask[p]) && abs(p - qp) <= HALF && slot >= 0 && slot < KEYS;
    float s[4] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY};
    if (ok) {
      const size_t r = ((size_t)w * 32 + i) * KEYS + slot;
      float x[16];
      if (zbf16) {
        const uint4* q = reinterpret_cast<const uint4*>(reinterpret_cast<const bf16*>(z) + r * 16);
#pragma unroll
        for (int k = 0; k < 2; ++k) {
          const uint4 u = q[k];
          const uint32_t v[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            x[8 * k + 2 * j] = __uint_as_float(v[j] << 16);
            x[8 * k + 2 * j + 1] = __uint_as_float(v[j] & 0xffff0000u);
          }
        }
      } else {
        const float4* q = reinterpret_cast<const float4*>(reinterpret_cast<const float*>(z) + r * 16);
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          const float4 u = q[k];
          x[4 * k] = u.x; x[4 * k + 1] = u.y; x[4 * k + 2] = u.z; x[4 * k + 3] = u.w;
        }
      }
      float mean = 0.f;
#pragma unroll
      for (int k = 0; k < 16; ++k) mean += x[k];
      mean *= (1.f / 16.f);
      float var = 0.f;
#pragma unroll
      for (int k = 0; k < 16; ++k) { const float d = x[k] - mean; var += d * d; }
      const float rstd = rsqrtf(var * (1.f / 16.f) + eps);
#pragma unroll
      for (int k = 0; k < 16; ++k) x[k] = (x[k] - mean) * rstd * gamma[k];
#pragma unroll
      for (int h = 0; h < 4; ++h) {
        float acc = 0.f;
#pragma unroll
        for (int k = 0; k < 16; ++k) acc = fmaf(x[k], wb[h * 16 + k], acc);
        s[h] = acc * LOG2E;
      }
    }
#pragma unroll
    for (int h = 0; h < 4; ++h) out[(size_t)h * count + e] = s[h];
  }
}

// ------------------------------------------------------------------------------------------------------------------ bf16
// dynamic smem: bias [64][BS] f32, then 2 stages of { Q [64][ST], K [192][ST], V [192][ST] } bf16
constexpr int E_BIAS = 64 * BS * 4;
constexpr int E_STAGE = (64 + 192 + 192) * ST * 2;
constexpr int E_SMEM = E_BIAS + 2 * E_STAGE;

extern "C" __global__ __launch_bounds__(128) void h_attn_esm(const bf16* __restrict__ q, const bf16* __restrict__ k, const bf16* __restrict__ v,
                                                              const float* __restrict__ bias, bf16* __restrict__ out, float* __restrict__ lse, int N, int A, int spc) {
  pdl_wait();
  extern __shared__ __align__(16) uint8_t sm[];
  float* sb = reinterpret_cast<float*>(sm);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const int q0 = 64 * blockIdx.x, h = blockIdx.y, a0 = blockIdx.z * spc, a1 = min(A, a0 + spc), nwin = N / 32, w0 = q0 / 32;
  // the warp's 144 positions start 16 warp rows into the staged K / V (which start at q0 - 64), at column 16 (warp & 1) of its window's bias
  const int kr0 = 16 * warp, bc0 = 16 * (warp & 1);
  load_bias(sb, bias, h, nwin, w0, tid);
  auto stage = [&](int s) { return su(sm + E_BIAS + (s & 1) * E_STAGE); };
  auto load = [&](int s, int a) {
    const uint32_t b = stage(s);
    load_rows(b, q, a, h, q0, 64, N, ST, tid);
    load_rows(b + 64 * ST * 2, k, a, h, q0 - HALF, 192, N, ST, tid);
    load_rows(b + (64 + 192) * ST * 2, v, a, h, q0 - HALF, 192, N, ST, tid);
  };
  if (a0 < a1) load(0, a0);
  cp_commit();
  for (int a = a0; a < a1; ++a) {
    const int s = (a - a0) & 1;
    if (a + 1 < a1) load(s ^ 1, a + 1);
    cp_commit();
    cp_wait<1>();
    __syncthreads();
    const uint32_t sQ = stage(s), sK = sQ + 64 * ST * 2, sV = sK + 192 * ST * 2;
    uint32_t aq[2][4];
#pragma unroll
    for (int ks = 0; ks < 2; ++ks) ldsm(aq[ks], sQ + ((16 * warp + (lane & 7) + ((lane >> 3) & 1) * 8) * ST + 16 * ks + (lane >> 4) * 8) * 2);
    float sc[WSPAN / 8][4];
#pragma unroll
    for (int jt = 0; jt < WSPAN / 8; ++jt) {
      uint32_t b[4];
      ldsm(b, sK + ((kr0 + 8 * jt + (lane & 7)) * ST + (lane >> 3) * 8) * 2);
      sc[jt][0] = sc[jt][1] = sc[jt][2] = sc[jt][3] = 0.f;
      mma(sc[jt], aq[0], b[0], b[1]);
      mma(sc[jt], aq[1], b[2], b[3]);
    }
    float m0 = -INFINITY, m1 = -INFINITY;
    const float* br = sb + (16 * warp + g) * BS + bc0 + 2 * t;
#pragma unroll
    for (int jt = 0; jt < WSPAN / 8; ++jt) {
      const float2 b0 = *reinterpret_cast<const float2*>(br + 8 * jt), b1 = *reinterpret_cast<const float2*>(br + 8 * BS + 8 * jt);
      sc[jt][0] = fmaf(sc[jt][0], C1, b0.x);
      sc[jt][1] = fmaf(sc[jt][1], C1, b0.y);
      sc[jt][2] = fmaf(sc[jt][2], C1, b1.x);
      sc[jt][3] = fmaf(sc[jt][3], C1, b1.y);
      m0 = fmaxf(m0, fmaxf(sc[jt][0], sc[jt][1]));
      m1 = fmaxf(m1, fmaxf(sc[jt][2], sc[jt][3]));
    }
    m0 = quad_max(m0);
    m1 = quad_max(m1);
    const float n0 = m0 == -INFINITY ? 0.f : m0, n1 = m1 == -INFINITY ? 0.f : m1;
    float l0 = 0.f, l1 = 0.f;
    uint32_t pp[WSPAN / 8][2];
#pragma unroll
    for (int jt = 0; jt < WSPAN / 8; ++jt) {
      const float p0 = ex2(sc[jt][0] - n0), p1 = ex2(sc[jt][1] - n0), p2 = ex2(sc[jt][2] - n1), p3 = ex2(sc[jt][3] - n1);
      l0 += p0 + p1;
      l1 += p2 + p3;
      pp[jt][0] = pk(p0, p1);
      pp[jt][1] = pk(p2, p3);
    }
    l0 = quad_sum(l0);
    l1 = quad_sum(l1);
    float o[4][4];
#pragma unroll
    for (int n = 0; n < 4; ++n) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.f;
#pragma unroll
    for (int ks = 0; ks < WSPAN / 16; ++ks) {
      const uint32_t ap[4] = {pp[2 * ks][0], pp[2 * ks][1], pp[2 * ks + 1][0], pp[2 * ks + 1][1]};
#pragma unroll
      for (int np = 0; np < 2; ++np) {
        uint32_t b[4];
        ldsmt(b, sV + ((kr0 + 16 * ks + ((lane >> 3) & 1) * 8 + (lane & 7)) * ST + 8 * (2 * np + (lane >> 4))) * 2);
        mma(o[2 * np], ap, b[0], b[1]);
        mma(o[2 * np + 1], ap, b[2], b[3]);
      }
    }
    const float i0 = l0 > 0.f ? 1.f / l0 : 0.f, i1 = l1 > 0.f ? 1.f / l1 : 0.f;
    float on[4][4];
#pragma unroll
    for (int n = 0; n < 4; ++n) { on[n][0] = o[n][0] * i0; on[n][1] = o[n][1] * i0; on[n][2] = o[n][2] * i1; on[n][3] = o[n][3] * i1; }
    // the warp's own Q rows (read only by this warp, already in registers) serve as the staging area
    store16x32(sm + E_BIAS + s * E_STAGE + warp * 16 * ST * 2, on, out + ((size_t)a * N + q0 + 16 * warp) * 128 + h * 32, 128, lane);
    if (lse != nullptr && t == 0) {   // an empty row (no key in its band) gets 1e30: its probabilities are 0 in the backward
      lse[((size_t)a * 4 + h) * N + q0 + 16 * warp + g] = l0 > 0.f ? n0 + __log2f(l0) : 1e30f;
      lse[((size_t)a * 4 + h) * N + q0 + 16 * warp + g + 8] = l1 > 0.f ? n1 + __log2f(l1) : 1e30f;
    }
    __syncthreads();
  }
}

// ------------------------------------------------------------------------------------------------------------------ TF32
// dynamic smem: bias [64][BS] f32, Q [64][ST32], K [192][ST32], V [192][ST32] f32
constexpr int E_SMEM32 = 64 * BS * 4 + (64 + 192 + 192) * ST32 * 4;

extern "C" __global__ __launch_bounds__(128) void h_attn_esm_tf32(const float* __restrict__ q, const float* __restrict__ k,
                                                                   const float* __restrict__ v, const float* __restrict__ bias,
                                                                   float* __restrict__ out, int N, int A, int spc) {
  pdl_wait();
  extern __shared__ __align__(16) uint8_t sm[];
  float* sb = reinterpret_cast<float*>(sm);
  float* sQ = sb + 64 * BS;
  float* sK = sQ + 64 * ST32;
  float* sV = sK + 192 * ST32;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const int q0 = 64 * blockIdx.x, h = blockIdx.y, a0 = blockIdx.z * spc, a1 = min(A, a0 + spc), nwin = N / 32, w0 = q0 / 32;
  const int kr0 = 16 * warp, bc0 = 16 * (warp & 1);
  load_bias(sb, bias, h, nwin, w0, tid);
  for (int a = a0; a < a1; ++a) {
    load_rows(su(sQ), q, a, h, q0, 64, N, ST32, tid);
    load_rows(su(sK), k, a, h, q0 - HALF, 192, N, ST32, tid);
    load_rows(su(sV), v, a, h, q0 - HALF, 192, N, ST32, tid);
    cp_commit();
    cp_wait<0>();
    __syncthreads();
    uint32_t aq[4][4];
#pragma unroll
    for (int kt = 0; kt < 4; ++kt) {
      const float* r0 = sQ + (16 * warp + g) * ST32 + 8 * kt + t;
      aq[kt][0] = __float_as_uint(r0[0]);
      aq[kt][1] = __float_as_uint(r0[8 * ST32]);
      aq[kt][2] = __float_as_uint(r0[4]);
      aq[kt][3] = __float_as_uint(r0[8 * ST32 + 4]);
    }
    float sc[WSPAN / 8][4];
#pragma unroll
    for (int nt = 0; nt < WSPAN / 8; ++nt) {
      sc[nt][0] = sc[nt][1] = sc[nt][2] = sc[nt][3] = 0.f;
      const float* kr = sK + (kr0 + 8 * nt + g) * ST32 + t;
#pragma unroll
      for (int kt = 0; kt < 4; ++kt) mma_tf32(sc[nt], aq[kt], __float_as_uint(kr[8 * kt]), __float_as_uint(kr[8 * kt + 4]));
    }
    float m0 = -INFINITY, m1 = -INFINITY;
    const float* br = sb + (16 * warp + g) * BS + bc0 + 2 * t;
#pragma unroll
    for (int nt = 0; nt < WSPAN / 8; ++nt) {
      const float2 b0 = *reinterpret_cast<const float2*>(br + 8 * nt), b1 = *reinterpret_cast<const float2*>(br + 8 * BS + 8 * nt);
      sc[nt][0] = fmaf(sc[nt][0], C1, b0.x);
      sc[nt][1] = fmaf(sc[nt][1], C1, b0.y);
      sc[nt][2] = fmaf(sc[nt][2], C1, b1.x);
      sc[nt][3] = fmaf(sc[nt][3], C1, b1.y);
      m0 = fmaxf(m0, fmaxf(sc[nt][0], sc[nt][1]));
      m1 = fmaxf(m1, fmaxf(sc[nt][2], sc[nt][3]));
    }
    m0 = quad_max(m0);
    m1 = quad_max(m1);
    const float n0 = m0 == -INFINITY ? 0.f : m0, n1 = m1 == -INFINITY ? 0.f : m1;
    float l0 = 0.f, l1 = 0.f;
    float o[4][4];
#pragma unroll
    for (int d = 0; d < 4; ++d) o[d][0] = o[d][1] = o[d][2] = o[d][3] = 0.f;
#pragma unroll
    for (int nt = 0; nt < WSPAN / 8; ++nt) {
      const float p0 = ex2(sc[nt][0] - n0), p1 = ex2(sc[nt][1] - n0), p2 = ex2(sc[nt][2] - n1), p3 = ex2(sc[nt][3] - n1);
      l0 += p0 + p1;
      l1 += p2 + p3;
      // the k slots permuted (slot t <-> key 8 nt + 2 t, slot t + 4 <-> 8 nt + 2 t + 1): the S accumulator is the A operand in place
      const uint32_t ap[4] = {tf(p0), tf(p2), tf(p1), tf(p3)};
      const float* vr = sV + (kr0 + 8 * nt + 2 * t) * ST32 + g;
#pragma unroll
      for (int d = 0; d < 4; ++d) mma_tf32(o[d], ap, __float_as_uint(vr[8 * d]), __float_as_uint(vr[ST32 + 8 * d]));
    }
    l0 = quad_sum(l0);
    l1 = quad_sum(l1);
    const float i0 = l0 > 0.f ? 1.f / l0 : 0.f, i1 = l1 > 0.f ? 1.f / l1 : 0.f;
    const int r0 = q0 + 16 * warp + g, r1 = r0 + 8;
#pragma unroll
    for (int d = 0; d < 4; ++d) {
      *reinterpret_cast<float2*>(out + ((size_t)a * N + r0) * 128 + h * 32 + 8 * d + 2 * t) = make_float2(o[d][0] * i0, o[d][1] * i0);
      *reinterpret_cast<float2*>(out + ((size_t)a * N + r1) * 128 + h * 32 + 8 * d + 2 * t) = make_float2(o[d][2] * i1, o[d][3] * i1);
    }
    __syncthreads();
  }
}

// ------------------------------------------------------------------------------------------------------------------ backward (bf16)
// dQ and the pair-bias gradient. dynamic smem: bias [64][BS] f32, then { Q, dO [64][ST]; K, V [192][ST] bf16; LSE, D [64] f32 } (one stage)
constexpr int Q_STAGE = (64 * 2 + 192 * 2) * ST * 2 + 2 * 64 * 4;
constexpr int Q_SMEM = E_BIAS + Q_STAGE;

extern "C" __global__ __launch_bounds__(128) void h_attn_esm_dq(const bf16* __restrict__ q, const bf16* __restrict__ k, const bf16* __restrict__ v,
                                                                 const bf16* __restrict__ dout, const float* __restrict__ bias,
                                                                 const float* __restrict__ lse, const float* __restrict__ dd, bf16* __restrict__ out,
                                                                 float* __restrict__ partial, int ldd, int N, int A, int spc) {
  pdl_wait();
  extern __shared__ __align__(16) uint8_t sm[];
  float* sb = reinterpret_cast<float*>(sm);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const int q0 = 64 * blockIdx.x, h = blockIdx.y, a0 = blockIdx.z * spc, a1 = min(A, a0 + spc), nwin = N / 32, w0 = q0 / 32;
  const int kr0 = 16 * warp, bc0 = 16 * (warp & 1);
  load_bias(sb, bias, h, nwin, w0, tid);
  uint8_t* st = sm + E_BIAS;
  const uint32_t sQ = su(st), sD = sQ + 64 * ST * 2, sK = sD + 64 * ST * 2, sV = sK + 192 * ST * 2;
  float* sl = reinterpret_cast<float*>(st + (64 * 2 + 192 * 2) * ST * 2);
  float dba[WSPAN / 8][4];
#pragma unroll
  for (int i = 0; i < WSPAN / 8; ++i) dba[i][0] = dba[i][1] = dba[i][2] = dba[i][3] = 0.f;
  for (int a = a0; a < a1; ++a) {
    load_rows(sQ, q, a, h, q0, 64, N, ST, tid);
    load_rows(sD, dout, a, h, q0, 64, N, ST, tid);
    load_rows(sK, k, a, h, q0 - HALF, 192, N, ST, tid);
    load_rows(sV, v, a, h, q0 - HALF, 192, N, ST, tid);
    load_vec(sl, lse, a, h, q0, 64, N, tid);
    load_vec(sl + 64, dd, a, h, q0, 64, N, tid);
    cp_commit();
    cp_wait<0>();
    __syncthreads();
    uint32_t aq[2][4], ad[2][4];
#pragma unroll
    for (int ks = 0; ks < 2; ++ks) {
      const uint32_t off = ((16 * warp + (lane & 7) + ((lane >> 3) & 1) * 8) * ST + 16 * ks + (lane >> 4) * 8) * 2;
      ldsm(aq[ks], sQ + off);
      ldsm(ad[ks], sD + off);
    }
    const float lse0 = sl[16 * warp + g], lse1 = sl[16 * warp + g + 8], d0 = sl[64 + 16 * warp + g], d1 = sl[64 + 16 * warp + g + 8];
    float dq[4][4];
#pragma unroll
    for (int n = 0; n < 4; ++n) dq[n][0] = dq[n][1] = dq[n][2] = dq[n][3] = 0.f;
    const float* br = sb + (16 * warp + g) * BS + bc0 + 2 * t;
#pragma unroll
    for (int grp = 0; grp < 3; ++grp) {
      float sc[6][4], dp[6][4];
#pragma unroll
      for (int i = 0; i < 6; ++i) {
        const int jt = 6 * grp + i;
        uint32_t b[4];
        ldsm(b, sK + ((kr0 + 8 * jt + (lane & 7)) * ST + (lane >> 3) * 8) * 2);
        sc[i][0] = sc[i][1] = sc[i][2] = sc[i][3] = 0.f;
        mma(sc[i], aq[0], b[0], b[1]);
        mma(sc[i], aq[1], b[2], b[3]);
        ldsm(b, sV + ((kr0 + 8 * jt + (lane & 7)) * ST + (lane >> 3) * 8) * 2);
        dp[i][0] = dp[i][1] = dp[i][2] = dp[i][3] = 0.f;
        mma(dp[i], ad[0], b[0], b[1]);
        mma(dp[i], ad[1], b[2], b[3]);
      }
      uint32_t ds[6][2];
#pragma unroll
      for (int i = 0; i < 6; ++i) {
        const int jt = 6 * grp + i;
        const float2 b0 = *reinterpret_cast<const float2*>(br + 8 * jt), b1 = *reinterpret_cast<const float2*>(br + 8 * BS + 8 * jt);
        const float p0 = ex2(fmaf(sc[i][0], C1, b0.x) - lse0), p1 = ex2(fmaf(sc[i][1], C1, b0.y) - lse0);
        const float p2 = ex2(fmaf(sc[i][2], C1, b1.x) - lse1), p3 = ex2(fmaf(sc[i][3], C1, b1.y) - lse1);
        const float s0 = p0 * (dp[i][0] - d0), s1 = p1 * (dp[i][1] - d0), s2 = p2 * (dp[i][2] - d1), s3 = p3 * (dp[i][3] - d1);
        ds[i][0] = pk(s0, s1);
        ds[i][1] = pk(s2, s3);
        dba[jt][0] += s0;
        dba[jt][1] += s1;
        dba[jt][2] += s2;
        dba[jt][3] += s3;
      }
#pragma unroll
      for (int ks = 0; ks < 3; ++ks) {
        const uint32_t ap[4] = {ds[2 * ks][0], ds[2 * ks][1], ds[2 * ks + 1][0], ds[2 * ks + 1][1]};
#pragma unroll
        for (int np = 0; np < 2; ++np) {
          uint32_t b[4];
          ldsmt(b, sK + ((kr0 + 16 * (3 * grp + ks) + ((lane >> 3) & 1) * 8 + (lane & 7)) * ST + 8 * (2 * np + (lane >> 4))) * 2);
          mma(dq[2 * np], ap, b[0], b[1]);
          mma(dq[2 * np + 1], ap, b[2], b[3]);
        }
      }
    }
    // the warp's own Q rows (read only by this warp, already in registers) serve as the staging area
    store16x32(st + warp * 16 * ST * 2, dq, out + ((size_t)a * N + q0 + 16 * warp) * ldd + h * 32, ldd, lane, SCALE);
    __syncthreads();
  }
  // dbias partial of this sample chunk, natural units, layout [h][w][i][c]
  float* pz = partial + (size_t)blockIdx.z * 4 * nwin * 32 * SPAN;
  const int w = w0 + (warp >> 1), i0 = 16 * (warp & 1) + g;
#pragma unroll
  for (int jt = 0; jt < WSPAN / 8; ++jt) {
    const int c = bc0 + 8 * jt + 2 * t;
    *reinterpret_cast<float2*>(pz + ((size_t)(h * nwin + w) * 32 + i0) * SPAN + c) = make_float2(dba[jt][0], dba[jt][1]);
    *reinterpret_cast<float2*>(pz + ((size_t)(h * nwin + w) * 32 + i0 + 8) * SPAN + c) = make_float2(dba[jt][2], dba[jt][3]);
  }
}

// dK and dV. dynamic smem: bias band per warp [4][16 keys][BSW] f32 (query column c of warp w is the query qs + 16 w + c, -inf outside the
// band), then { Q, dO [192][ST]; K, V [64][ST] bf16; LSE, D [192] f32 }
constexpr int BSW = 164;   // = 4 mod 32: float2 reads at 4 g + 2 t banks (2-way)
constexpr int K_BIAS = 4 * 16 * BSW * 4;
constexpr int K_STAGE = (192 * 2 + 64 * 2) * ST * 2 + 2 * 192 * 4;
constexpr int K_SMEM = K_BIAS + K_STAGE;

extern "C" __global__ __launch_bounds__(128) void h_attn_esm_dkv(const bf16* __restrict__ q, const bf16* __restrict__ k, const bf16* __restrict__ v,
                                                                  const bf16* __restrict__ dout, const float* __restrict__ bias,
                                                                  const float* __restrict__ lse, const float* __restrict__ dd, bf16* __restrict__ out,
                                                                  int ldd, int N, int A, int spc) {
  pdl_wait();
  extern __shared__ __align__(16) uint8_t sm[];
  float* sb = reinterpret_cast<float*>(sm);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const int k0 = 64 * blockIdx.x, qs = k0 - 64, h = blockIdx.y, a0 = blockIdx.z * spc, a1 = min(A, a0 + spc), nwin = N / 32;
  const int kr = 16 * warp;
  for (int i = tid; i < 4 * 16 * BSW; i += 128) sb[i] = -INFINITY;
  __syncthreads();
  for (int i = tid; i < 64 * WSPAN; i += 128) {   // key fastest: the bias rows are contiguous in the key position
    const int kk = i & 63, qc = i >> 6, w = kk >> 4;
    const int qq = qs + 16 * w + qc, key = k0 + kk;
    if (qq < 0 || qq >= N || abs(qq - key) > HALF) continue;
    const int win = qq >> 5;
    sb[(w * 16 + (kk & 15)) * BSW + qc] = bias[((size_t)(h * nwin + win) * 32 + (qq & 31)) * SPAN + (key - (32 * win - HALF))];
  }
  uint8_t* st = sm + K_BIAS;
  const uint32_t sQ = su(st), sD = sQ + 192 * ST * 2, sK = sD + 192 * ST * 2, sV = sK + 64 * ST * 2;
  float* sl = reinterpret_cast<float*>(st + (192 * 2 + 64 * 2) * ST * 2);
  for (int a = a0; a < a1; ++a) {
    load_rows(sQ, q, a, h, qs, 192, N, ST, tid);
    load_rows(sD, dout, a, h, qs, 192, N, ST, tid);
    load_rows(sK, k, a, h, k0, 64, N, ST, tid);
    load_rows(sV, v, a, h, k0, 64, N, ST, tid);
    load_vec(sl, lse, a, h, qs, 192, N, tid);
    load_vec(sl + 192, dd, a, h, qs, 192, N, tid);
    cp_commit();
    cp_wait<0>();
    __syncthreads();
    uint32_t ak[2][4], av[2][4];
#pragma unroll
    for (int ks = 0; ks < 2; ++ks) {
      const uint32_t off = ((kr + (lane & 7) + ((lane >> 3) & 1) * 8) * ST + 16 * ks + (lane >> 4) * 8) * 2;
      ldsm(ak[ks], sK + off);
      ldsm(av[ks], sV + off);
    }
    float dk[4][4], dv[4][4];
#pragma unroll
    for (int n = 0; n < 4; ++n) dk[n][0] = dk[n][1] = dk[n][2] = dk[n][3] = dv[n][0] = dv[n][1] = dv[n][2] = dv[n][3] = 0.f;
#pragma unroll
    for (int grp = 0; grp < 3; ++grp) {
      float sc[6][4], dp[6][4];
#pragma unroll
      for (int i = 0; i < 6; ++i) {
        const int qr = kr + 48 * grp + 8 * i;
        uint32_t b[4];
        ldsm(b, sQ + ((qr + (lane & 7)) * ST + (lane >> 3) * 8) * 2);
        sc[i][0] = sc[i][1] = sc[i][2] = sc[i][3] = 0.f;
        mma(sc[i], ak[0], b[0], b[1]);
        mma(sc[i], ak[1], b[2], b[3]);
        ldsm(b, sD + ((qr + (lane & 7)) * ST + (lane >> 3) * 8) * 2);
        dp[i][0] = dp[i][1] = dp[i][2] = dp[i][3] = 0.f;
        mma(dp[i], av[0], b[0], b[1]);
        mma(dp[i], av[1], b[2], b[3]);
      }
      uint32_t pt[6][2], dst[6][2];
#pragma unroll
      for (int i = 0; i < 6; ++i) {
        const int c0 = kr + 48 * grp + 8 * i + 2 * t;
        const float2 lq = *reinterpret_cast<const float2*>(sl + c0), dq2 = *reinterpret_cast<const float2*>(sl + 192 + c0);
        const float* bk0 = sb + (warp * 16 + g) * BSW + (c0 - kr);
        const float* bk1 = bk0 + 8 * BSW;
        const float p0 = ex2(fmaf(sc[i][0], C1, bk0[0]) - lq.x), p1 = ex2(fmaf(sc[i][1], C1, bk0[1]) - lq.y);
        const float p2 = ex2(fmaf(sc[i][2], C1, bk1[0]) - lq.x), p3 = ex2(fmaf(sc[i][3], C1, bk1[1]) - lq.y);
        pt[i][0] = pk(p0, p1);
        pt[i][1] = pk(p2, p3);
        dst[i][0] = pk(p0 * (dp[i][0] - dq2.x), p1 * (dp[i][1] - dq2.y));
        dst[i][1] = pk(p2 * (dp[i][2] - dq2.x), p3 * (dp[i][3] - dq2.y));
      }
#pragma unroll
      for (int ks = 0; ks < 3; ++ks) {
        const uint32_t ap[4] = {pt[2 * ks][0], pt[2 * ks][1], pt[2 * ks + 1][0], pt[2 * ks + 1][1]};
        const uint32_t as[4] = {dst[2 * ks][0], dst[2 * ks][1], dst[2 * ks + 1][0], dst[2 * ks + 1][1]};
        const int qrow = kr + 48 * grp + 16 * ks + ((lane >> 3) & 1) * 8 + (lane & 7);
#pragma unroll
        for (int np = 0; np < 2; ++np) {
          uint32_t b[4];
          const int cc = 8 * (2 * np + (lane >> 4));
          ldsmt(b, sD + (qrow * ST + cc) * 2);
          mma(dv[2 * np], ap, b[0], b[1]);
          mma(dv[2 * np + 1], ap, b[2], b[3]);
          ldsmt(b, sQ + (qrow * ST + cc) * 2);
          mma(dk[2 * np], as, b[0], b[1]);
          mma(dk[2 * np + 1], as, b[2], b[3]);
        }
      }
    }
    uint8_t* kstg = st + (192 * 2) * ST * 2 + kr * ST * 2;   // this warp's own K / V rows (read only by it, already in registers)
    store16x32(kstg, dk, out + ((size_t)a * N + k0 + kr) * ldd + h * 32 + 128, ldd, lane, SCALE);
    store16x32(kstg + 64 * ST * 2, dv, out + ((size_t)a * N + k0 + kr) * ldd + h * 32 + 256, ldd, lane);
    __syncthreads();
  }
}

// ------------------------------------------------------------------------------------------------------------------ pair bias backward
// z: the trunked pair [nwin, 32, 192, 16] bf16 / fp32; dbias [4][nwin][32][160] fp32 (natural units, by position, as the forward lays it out);
// dz in z's dtype (0 on the slots no position of the band maps to); part[block][h * 16 + c] = sum dbias[h] LN(z)[c] for local_bias_fin.
extern "C" __global__ __launch_bounds__(256) void h_pair_bias_esm_bwd(const void* __restrict__ z, const float* __restrict__ wb, const float* __restrict__ gamma,
                                                                       const float* __restrict__ dbias, const int* __restrict__ nreal, void* __restrict__ dz,
                                                                       float* __restrict__ part, int N, float eps, int zbf16) {
  pdl_wait();
  __shared__ float sw[64], red[8][64];
  if (threadIdx.x < 64) sw[threadIdx.x] = wb[threadIdx.x] * gamma[threadIdx.x & 15];
  __syncthreads();
  const int nwin = N / 32, nr = *nreal;
  const size_t rows = (size_t)nwin * 32 * KEYS;
  float G[64];
#pragma unroll
  for (int i = 0; i < 64; ++i) G[i] = 0.f;
  for (size_t r = (size_t)blockIdx.x * blockDim.x + threadIdx.x; r < rows; r += (size_t)gridDim.x * blockDim.x) {
    const int slot = (int)(r % KEYS), i = (int)((r / KEYS) % 32), w = (int)(r / (32 * KEYS));
    const int first = 32 * w - (KEYS - 32) / 2;
    const int start = first < 0 ? 0 : min(first, nr - KEYS);
    const int p = start + slot, qp = 32 * w + i, c = p - (32 * w - HALF);
    float db[4] = {0.f, 0.f, 0.f, 0.f};
    const bool live = p >= 0 && p < N && p < nr && abs(p - qp) <= HALF && c >= 0 && c < SPAN;
    if (live) {
#pragma unroll
      for (int h = 0; h < 4; ++h) db[h] = dbias[((size_t)(h * nwin + w) * 32 + i) * SPAN + c];
    }
    float x[16];
    if (zbf16) {
      const uint4* pz = reinterpret_cast<const uint4*>(reinterpret_cast<const bf16*>(z) + r * 16);
#pragma unroll
      for (int k = 0; k < 2; ++k) {
        const uint4 u = pz[k];
        const uint32_t v[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
        for (int j = 0; j < 4; ++j) { x[8 * k + 2 * j] = __uint_as_float(v[j] << 16); x[8 * k + 2 * j + 1] = __uint_as_float(v[j] & 0xffff0000u); }
      }
    } else {
      const float4* pz = reinterpret_cast<const float4*>(reinterpret_cast<const float*>(z) + r * 16);
#pragma unroll
      for (int k = 0; k < 4; ++k) { const float4 u = pz[k]; x[4 * k] = u.x; x[4 * k + 1] = u.y; x[4 * k + 2] = u.z; x[4 * k + 3] = u.w; }
    }
    float mean = 0.f;
#pragma unroll
    for (int k = 0; k < 16; ++k) mean += x[k];
    mean *= 0.0625f;
    float var = 0.f;
#pragma unroll
    for (int k = 0; k < 16; ++k) { x[k] -= mean; var += x[k] * x[k]; }
    const float rstd = 1.f / sqrtf(var * 0.0625f + eps);
#pragma unroll
    for (int k = 0; k < 16; ++k) x[k] *= rstd;
    float dzh[16];
#pragma unroll
    for (int k = 0; k < 16; ++k) {
      dzh[k] = db[0] * sw[k] + db[1] * sw[16 + k] + db[2] * sw[32 + k] + db[3] * sw[48 + k];
#pragma unroll
      for (int h = 0; h < 4; ++h) G[h * 16 + k] += db[h] * x[k];
    }
    float m1 = 0.f, m2 = 0.f;
#pragma unroll
    for (int k = 0; k < 16; ++k) { m1 += dzh[k]; m2 += dzh[k] * x[k]; }
    m1 *= 0.0625f; m2 *= 0.0625f;
    float o[16];
#pragma unroll
    for (int k = 0; k < 16; ++k) o[k] = rstd * (dzh[k] - m1 - x[k] * m2);
    if (zbf16) {
      uint32_t wv[8];
#pragma unroll
      for (int k = 0; k < 8; ++k) wv[k] = pk(o[2 * k], o[2 * k + 1]);
      uint4* pd = reinterpret_cast<uint4*>(reinterpret_cast<bf16*>(dz) + r * 16);
      pd[0] = make_uint4(wv[0], wv[1], wv[2], wv[3]);
      pd[1] = make_uint4(wv[4], wv[5], wv[6], wv[7]);
    } else {
      float4* pd = reinterpret_cast<float4*>(reinterpret_cast<float*>(dz) + r * 16);
#pragma unroll
      for (int k = 0; k < 4; ++k) pd[k] = make_float4(o[4 * k], o[4 * k + 1], o[4 * k + 2], o[4 * k + 3]);
    }
  }
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
#pragma unroll
  for (int i = 0; i < 64; ++i) {
    float v = G[i];
#pragma unroll
    for (int o = 16; o >= 1; o >>= 1) v += __shfl_xor_sync(~0u, v, o);
    if (lane == 0) red[warp][i] = v;
  }
  __syncthreads();
  if (threadIdx.x < 64) {
    float s = 0.f;
#pragma unroll
    for (int w = 0; w < 8; ++w) s += red[w][threadIdx.x];
    part[(size_t)blockIdx.x * 64 + threadIdx.x] = s;
  }
}
