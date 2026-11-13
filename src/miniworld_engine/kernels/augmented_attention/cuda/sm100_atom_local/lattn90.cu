// lattn90.cu — AF3 windowed atom attention (32 queries x 128 keys per window, per-window pair bias) on sm_90a with mma.sync.
//
//   S = q k^T / sqrt(32) + bias[h][w]      O = softmax(S) v      (bf16 O [A N, 128], row LSE [A, 4, N] in log2 units)
//
// Plain warp-level tensor-core code: the 32-wide heads and 128-key windows are softmax/latency bound, not MMA bound, so many small
// warps per SM (and cp.async double buffering inside a CTA that walks a chunk of the samples) hide the latency that the one-warpgroup
// wgmma kernels of lattn_{fwd,dq,dkv}.cu expose. The pair bias (plus key mask, log2 domain) is staged in shared memory once per CTA and
// reused by every sample the CTA walks; dbias is accumulated in registers over those samples (one partial per sample chunk).
//
//   h_bias_prep  bias2 = bias log2(e), -inf where the key is outside [0, N) or masked.
//   h_attn_fwd   CTA = 64 queries (2 windows) x 1 head x a chunk of samples; warp = 16 queries; K / V = the 160-key union.
//   h_attn_dq    same geometry, dQ.
//   h_attn_dkv   CTA = 64 keys x 1 head x a chunk of samples; warp = 16 keys; Q / dO = the 192-query union; dK, dV, dbias partial.
// SPDX-License-Identifier: Apache-2.0
#include <cuda_bf16.h>
#include <stdint.h>
#define DEVI __device__ __forceinline__

typedef __nv_bfloat16 bf16;

namespace {
constexpr int ST = 40;      // bf16 row stride of the staged tiles (80 B: conflict-free ldmatrix)
constexpr int BSQ = 136;    // fp32 row stride of the query-centric bias tile
constexpr int BSK = 193;    // fp32 row stride of the key-centric bias tile
constexpr float SCALE = 0.17677669529663687f;
constexpr float LOG2E = 1.4426950408889634f;
constexpr float C1 = SCALE * LOG2E;

DEVI void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
DEVI void pdl_trigger() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }
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
DEVI uint32_t pk(float lo, float hi) {
  uint32_t r;
  asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(r) : "f"(lo), "f"(hi));
  return r;
}
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
// rows [r0, r0 + n) of head h of x[a] (x: [A, N, 128] bf16) into shared rows of stride ST; rows outside [0, N) are zeros
DEVI void load_rows(uint32_t dst, const bf16* x, int a, int h, int r0, int n, int N, int tid) {
  for (int i = tid; i < n * 4; i += 128) {
    const int r = i >> 2, c = i & 3, gr = r0 + r;
    const bool ok = gr >= 0 && gr < N;
    cp16(dst + (r * ST + c * 8) * 2, x + ((size_t)a * N + (ok ? gr : 0)) * 128 + h * 32 + c * 8, ok);
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

// one warp's 16 x 32 bf16 result tile (C fragments d[4][4], times scale) -> global rows of ld elements through the warp's private
// shared rows (stride 80 B): 16-byte vector stores, each row segment a whole 64 B (two full sectors) instead of 16-byte pieces
DEVI void store16x32(uint8_t* stg, const float (*d)[4], float scale, bf16* row0, size_t ld, int lane) {
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

extern "C" __global__ void h_bias_prep(const float* __restrict__ bias, const uint8_t* __restrict__ mask, float* __restrict__ out, int N) {
  pdl_wait();
  const size_t count = (size_t)4 * (N / 32) * 32 * 128;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < count; i += (size_t)gridDim.x * blockDim.x) {
    const int w = (int)((i >> 12) % (N / 32)), key = 32 * w - 48 + (int)(i & 127);
    const bool ok = key >= 0 && key < N && (mask == nullptr || mask[key]);
    out[i] = ok ? bias[i] * LOG2E : -INFINITY;
  }
}

// ------------------------------------------------------------------------------------------------------------------ forward
// dynamic smem: bias [64][BSQ] f32, then 2 stages of { Q [64][ST], K [160][ST], V [160][ST] } bf16
constexpr int F_BIAS = 64 * BSQ * 4;
constexpr int F_STAGE = (64 + 160 + 160) * ST * 2;
// cp.async stages of the per-sample tiles: 1 = three CTAs per SM, 2 = double buffered (two CTAs). The forward keeps two (few samples per
// CTA at A <= 8: the prefetch is what hides the first load), dQ one (measured 236 -> 192 us at A48).
constexpr int FSTG = 2, QSTG = 1;
constexpr int F_SMEM = F_BIAS + FSTG * F_STAGE;

extern "C" __global__ __launch_bounds__(128) void h_attn_fwd(const bf16* __restrict__ q, const bf16* __restrict__ k, const bf16* __restrict__ v,
                                                              const float* __restrict__ bias, bf16* __restrict__ out, float* __restrict__ lse,
                                                              int N, int A, int spc, int G) {
  pdl_wait();
  extern __shared__ __align__(16) uint8_t sm[];
  float* sb = reinterpret_cast<float*>(sm);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  // G > 1: G distinct pairs (trunk_seeds), the samples laid out [A / G][G] (sample = j * G + group) and one bias [4, nwin, 32, 128] per group;
  // a CTA walks the samples of ONE group (stride G), so its bias tile still serves its whole chunk. G = 1 is the one-bias kernel.
  const int grp = G > 1 ? blockIdx.z % G : 0, chunk = G > 1 ? blockIdx.z / G : blockIdx.z;
  const int Ag = A / G, j0 = chunk * spc, nj = min(Ag, j0 + spc) - j0, a0 = j0 * G + grp;
  const int q0 = 64 * blockIdx.x, h = blockIdx.y, nwin = N / 32, w0 = q0 / 32;
  const int win = warp >> 1;
  bias += (size_t)grp * 4 * nwin * 32 * 128;
  // bias of the CTA's two windows [64 rows][128] -> padded rows
  for (int i = tid; i < 64 * 32; i += 128) {
    const int r = i >> 5, c = i & 31;
    cp16f(su(sb + r * BSQ + c * 4), bias + ((size_t)(h * nwin + w0) * 32 + r) * 128 + c * 4);
  }
  auto stage = [&](int s) { return su(sm + F_BIAS + (s % FSTG) * F_STAGE); };
  auto load = [&](int s, int a) {
    const uint32_t b = stage(s);
    load_rows(b, q, a, h, q0, 64, N, tid);
    load_rows(b + 64 * ST * 2, k, a, h, q0 - 48, 160, N, tid);
    load_rows(b + (64 + 160) * ST * 2, v, a, h, q0 - 48, 160, N, tid);
  };
  if (nj > 0) load(0, a0);
  cp_commit();
  for (int j = 0; j < nj; ++j) {
    const int a = a0 + j * G;
    const int s = j & 1;
    if (FSTG == 1) {
      if (j > 0) load(0, a);
      cp_commit();
      cp_wait<0>();
    } else {
      if (j + 1 < nj) load(s ^ 1, a + G);
      cp_commit();
      cp_wait<1>();
    }
    __syncthreads();
    const uint32_t sQ = stage(s), sK = sQ + 64 * ST * 2, sV = sK + 160 * ST * 2;
    uint32_t aq[2][4];
#pragma unroll
    for (int ks = 0; ks < 2; ++ks) ldsm(aq[ks], sQ + ((16 * warp + (lane & 7) + ((lane >> 3) & 1) * 8) * ST + 16 * ks + (lane >> 4) * 8) * 2);
    float sc[16][4];
#pragma unroll
    for (int jt = 0; jt < 16; ++jt) {
      uint32_t b[4];
      ldsm(b, sK + ((32 * win + 8 * jt + (lane & 7)) * ST + (lane >> 3) * 8) * 2);
      sc[jt][0] = sc[jt][1] = sc[jt][2] = sc[jt][3] = 0.f;
      mma(sc[jt], aq[0], b[0], b[1]);
      mma(sc[jt], aq[1], b[2], b[3]);
    }
    float m0 = -INFINITY, m1 = -INFINITY;
    const float* br = sb + (16 * warp + g) * BSQ + 2 * t;
#pragma unroll
    for (int jt = 0; jt < 16; ++jt) {
      const float2 b0 = *reinterpret_cast<const float2*>(br + 8 * jt), b1 = *reinterpret_cast<const float2*>(br + 8 * BSQ + 8 * jt);
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
    uint32_t pp[16][2];
#pragma unroll
    for (int jt = 0; jt < 16; ++jt) {
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
    for (int ks = 0; ks < 8; ++ks) {
      const uint32_t ap[4] = {pp[2 * ks][0], pp[2 * ks][1], pp[2 * ks + 1][0], pp[2 * ks + 1][1]};
#pragma unroll
      for (int np = 0; np < 2; ++np) {
        uint32_t b[4];
        ldsmt(b, sV + ((32 * win + 16 * ks + ((lane >> 3) & 1) * 8 + (lane & 7)) * ST + 8 * (2 * np + (lane >> 4))) * 2);
        mma(o[2 * np], ap, b[0], b[1]);
        mma(o[2 * np + 1], ap, b[2], b[3]);
      }
    }
    const float i0 = l0 > 0.f ? 1.f / l0 : 0.f, i1 = l1 > 0.f ? 1.f / l1 : 0.f;
    const int r0 = q0 + 16 * warp + g, r1 = r0 + 8;
    {
      float on[4][4];
#pragma unroll
      for (int n = 0; n < 4; ++n) { on[n][0] = o[n][0] * i0; on[n][1] = o[n][1] * i0; on[n][2] = o[n][2] * i1; on[n][3] = o[n][3] * i1; }
      // the warp's own Q rows (read only by this warp, already in registers) serve as the staging area
      store16x32(sm + F_BIAS + (j % FSTG) * F_STAGE + warp * 16 * ST * 2, on, 1.f, out + ((size_t)a * N + q0 + 16 * warp) * 128 + h * 32, 128, lane);
    }
    if (t == 0) {
      lse[((size_t)a * 4 + h) * N + r0] = l0 > 0.f ? n0 + __log2f(l0) : 1e30f;
      lse[((size_t)a * 4 + h) * N + r1] = l1 > 0.f ? n1 + __log2f(l1) : 1e30f;
    }
    __syncthreads();
  }
}

// ----------------------------------------------------------------------------------------------------------------- dQ
// dynamic smem: bias [64][BSQ] f32, then 2 stages of { Q, dO [64][ST]; K, V [160][ST] bf16; LSE, D [64] f32 }
constexpr int Q_STAGE = (64 * 2 + 160 * 2) * ST * 2 + 2 * 64 * 4;
constexpr int Q_SMEM = F_BIAS + QSTG * Q_STAGE;

extern "C" __global__ __launch_bounds__(128) void h_attn_dq(const bf16* __restrict__ q, const bf16* __restrict__ k, const bf16* __restrict__ v,
                                                             const bf16* __restrict__ dout, const float* __restrict__ bias,
                                                             const float* __restrict__ lse, const float* __restrict__ dd, bf16* __restrict__ out,
                                                             int ldd, int N, int A, int spc) {
  pdl_wait();
  extern __shared__ __align__(16) uint8_t sm[];
  float* sb = reinterpret_cast<float*>(sm);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const int q0 = 64 * blockIdx.x, h = blockIdx.y, a0 = blockIdx.z * spc, a1 = min(A, a0 + spc), nwin = N / 32, w0 = q0 / 32;
  const int win = warp >> 1;
  for (int i = tid; i < 64 * 32; i += 128) {
    const int r = i >> 5, c = i & 31;
    cp16f(su(sb + r * BSQ + c * 4), bias + ((size_t)(h * nwin + w0) * 32 + r) * 128 + c * 4);
  }
  auto stage = [&](int s) { return sm + F_BIAS + (s % QSTG) * Q_STAGE; };
  auto load = [&](int s, int a) {
    uint8_t* b = stage(s);
    const uint32_t u = su(b);
    load_rows(u, q, a, h, q0, 64, N, tid);
    load_rows(u + 64 * ST * 2, dout, a, h, q0, 64, N, tid);
    load_rows(u + 128 * ST * 2, k, a, h, q0 - 48, 160, N, tid);
    load_rows(u + (128 + 160) * ST * 2, v, a, h, q0 - 48, 160, N, tid);
    float* sl = reinterpret_cast<float*>(b + (64 * 2 + 160 * 2) * ST * 2);
    load_vec(sl, lse, a, h, q0, 64, N, tid);
    load_vec(sl + 64, dd, a, h, q0, 64, N, tid);
  };
  if (a0 < a1) load(0, a0);
  cp_commit();
  for (int a = a0; a < a1; ++a) {
    const int s = (a - a0) & 1;
    if (QSTG == 1) {
      if (a > a0) load(0, a);
      cp_commit();
      cp_wait<0>();
    } else {
      if (a + 1 < a1) load(s ^ 1, a + 1);
      cp_commit();
      cp_wait<1>();
    }
    __syncthreads();
    const uint32_t sQ = su(stage(s)), sD = sQ + 64 * ST * 2, sK = sD + 64 * ST * 2, sV = sK + 160 * ST * 2;
    const float* sl = reinterpret_cast<const float*>(stage(s) + (64 * 2 + 160 * 2) * ST * 2);
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
    const float* br = sb + (16 * warp + g) * BSQ + 2 * t;
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      float sc[8][4], dp[8][4];
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const int jt = 8 * half + i;
        uint32_t b[4];
        ldsm(b, sK + ((32 * win + 8 * jt + (lane & 7)) * ST + (lane >> 3) * 8) * 2);
        sc[i][0] = sc[i][1] = sc[i][2] = sc[i][3] = 0.f;
        mma(sc[i], aq[0], b[0], b[1]);
        mma(sc[i], aq[1], b[2], b[3]);
        ldsm(b, sV + ((32 * win + 8 * jt + (lane & 7)) * ST + (lane >> 3) * 8) * 2);
        dp[i][0] = dp[i][1] = dp[i][2] = dp[i][3] = 0.f;
        mma(dp[i], ad[0], b[0], b[1]);
        mma(dp[i], ad[1], b[2], b[3]);
      }
      uint32_t ds[8][2];
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const int jt = 8 * half + i;
        const float2 b0 = *reinterpret_cast<const float2*>(br + 8 * jt), b1 = *reinterpret_cast<const float2*>(br + 8 * BSQ + 8 * jt);
        const float p0 = ex2(fmaf(sc[i][0], C1, b0.x) - lse0), p1 = ex2(fmaf(sc[i][1], C1, b0.y) - lse0);
        const float p2 = ex2(fmaf(sc[i][2], C1, b1.x) - lse1), p3 = ex2(fmaf(sc[i][3], C1, b1.y) - lse1);
        ds[i][0] = pk(p0 * (dp[i][0] - d0), p1 * (dp[i][1] - d0));
        ds[i][1] = pk(p2 * (dp[i][2] - d1), p3 * (dp[i][3] - d1));
      }
#pragma unroll
      for (int ks = 0; ks < 4; ++ks) {
        const uint32_t ap[4] = {ds[2 * ks][0], ds[2 * ks][1], ds[2 * ks + 1][0], ds[2 * ks + 1][1]};
#pragma unroll
        for (int np = 0; np < 2; ++np) {
          uint32_t b[4];
          ldsmt(b, sK + ((32 * win + 64 * half + 16 * ks + ((lane >> 3) & 1) * 8 + (lane & 7)) * ST + 8 * (2 * np + (lane >> 4))) * 2);
          mma(dq[2 * np], ap, b[0], b[1]);
          mma(dq[2 * np + 1], ap, b[2], b[3]);
        }
      }
    }
    const int r0 = q0 + 16 * warp + g, r1 = r0 + 8;
    store16x32(const_cast<uint8_t*>(stage(s)) + warp * 16 * ST * 2, dq, SCALE, out + ((size_t)a * N + q0 + 16 * warp) * ldd + h * 32, ldd, lane);
    __syncthreads();
  }
}

// ----------------------------------------------------------------------------------------------------------------- dK, dV, dbias
// dynamic smem: bias^T [64 keys][BSK] f32 (-inf outside the band), then { Q, dO [192][ST]; K, V [64][ST] bf16; LSE, D [192] f32 }
constexpr int BSW = 132;   // per-warp band row stride: float2 reads, 4g + 2t banks (2-way)
constexpr int K_BIAS = 4 * 16 * BSW * 4;
constexpr int K_STAGE = (192 * 2 + 64 * 2) * ST * 2 + 2 * 192 * 4;
constexpr int K_SMEM = K_BIAS + K_STAGE;

extern "C" __global__ __launch_bounds__(128, 3) void h_attn_dkv(const bf16* __restrict__ q, const bf16* __restrict__ k, const bf16* __restrict__ v,
                                                              const bf16* __restrict__ dout, const float* __restrict__ bias,
                                                              const float* __restrict__ lse, const float* __restrict__ dd, bf16* __restrict__ out,
                                                              float* __restrict__ partial, int ldd, int N, int A, int spc) {
  pdl_wait();
  extern __shared__ __align__(16) uint8_t sm[];
  float* sb = reinterpret_cast<float*>(sm);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const int m = blockIdx.x, k0 = 64 * m, qs = k0 - 64, h = blockIdx.y, a0 = blockIdx.z * spc, a1 = min(A, a0 + spc), nwin = N / 32;
  const int kr = 16 * warp, qoff = 32 * (((warp - 3) >> 1) + 2);
  // band of the bias, per warp: sb[warp][key row 0..15][query column 0..127], query = qs + qoff(warp) + column
  for (int i = tid; i < 4 * 16 * BSW; i += 128) sb[i] = -INFINITY;
  __syncthreads();
#ifndef ABL_PRO
  for (int i = tid; i < 64 * 128; i += 128) {
    const int kk = i >> 7, qc = i & 127, w = kk >> 4;
    const int qq = qs + 32 * (((w - 3) >> 1) + 2) + qc;
    if (qq < 0 || qq >= N) continue;
    const int win = qq >> 5, j = k0 + kk - (32 * win - 48);
    if (j >= 0 && j < 128) sb[(w * 16 + (kk & 15)) * BSW + qc] = bias[((size_t)(h * nwin + win) * 32 + (qq & 31)) * 128 + j];
  }
#endif
  uint8_t* st = sm + K_BIAS;
  const uint32_t sQ = su(st), sD = sQ + 192 * ST * 2, sK = sD + 192 * ST * 2, sV = sK + 64 * ST * 2;
  float* sl = reinterpret_cast<float*>(st + (192 * 2 + 64 * 2) * ST * 2);
  float dba[16][4];
#pragma unroll
  for (int i = 0; i < 16; ++i) dba[i][0] = dba[i][1] = dba[i][2] = dba[i][3] = 0.f;
  for (int a = a0; a < a1; ++a) {
#ifndef ABL_LOAD
    load_rows(sQ, q, a, h, qs, 192, N, tid);
    load_rows(sD, dout, a, h, qs, 192, N, tid);
    load_rows(sK, k, a, h, k0, 64, N, tid);
    load_rows(sV, v, a, h, k0, 64, N, tid);
    load_vec(sl, lse, a, h, qs, 192, N, tid);
    load_vec(sl + 192, dd, a, h, qs, 192, N, tid);
#endif
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
    for (int half = 0; half < 2; ++half) {
      float sc[8][4], dp[8][4];
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const int qr = qoff + 64 * half + 8 * i;
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
      uint32_t pt[8][2], dst[8][2];
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const int c0 = qoff + 64 * half + 8 * i + 2 * t;
        const float2 lq = *reinterpret_cast<const float2*>(sl + c0), dq2 = *reinterpret_cast<const float2*>(sl + 192 + c0);
        const float* bk0 = sb + (warp * 16 + g) * BSW + (c0 - qoff);
        const float* bk1 = bk0 + 8 * BSW;
        const float p0 = ex2(fmaf(sc[i][0], C1, bk0[0]) - lq.x), p1 = ex2(fmaf(sc[i][1], C1, bk0[1]) - lq.y);
        const float p2 = ex2(fmaf(sc[i][2], C1, bk1[0]) - lq.x), p3 = ex2(fmaf(sc[i][3], C1, bk1[1]) - lq.y);
        const float s0 = p0 * (dp[i][0] - dq2.x), s1 = p1 * (dp[i][1] - dq2.y), s2 = p2 * (dp[i][2] - dq2.x), s3 = p3 * (dp[i][3] - dq2.y);
        pt[i][0] = pk(p0, p1);
        pt[i][1] = pk(p2, p3);
        dst[i][0] = pk(s0, s1);
        dst[i][1] = pk(s2, s3);
#ifndef ABL_DBA
        float* d = dba[8 * half + i];
        d[0] += s0;
        d[1] += s1;
        d[2] += s2;
        d[3] += s3;
#endif
      }
#ifndef ABL_PV
#pragma unroll
      for (int ks = 0; ks < 4; ++ks) {
        const uint32_t ap[4] = {pt[2 * ks][0], pt[2 * ks][1], pt[2 * ks + 1][0], pt[2 * ks + 1][1]};
        const uint32_t as[4] = {dst[2 * ks][0], dst[2 * ks][1], dst[2 * ks + 1][0], dst[2 * ks + 1][1]};
        const int qr = qoff + 64 * half + 16 * ks + ((lane >> 3) & 1) * 8 + (lane & 7);
#pragma unroll
        for (int np = 0; np < 2; ++np) {
          uint32_t b[4];
          const int cc = 8 * (2 * np + (lane >> 4));
          ldsmt(b, sD + (qr * ST + cc) * 2);
          mma(dv[2 * np], ap, b[0], b[1]);
          mma(dv[2 * np + 1], ap, b[2], b[3]);
          ldsmt(b, sQ + (qr * ST + cc) * 2);
          mma(dk[2 * np], as, b[0], b[1]);
          mma(dk[2 * np + 1], as, b[2], b[3]);
        }
      }
#else
      if (pt[0][0] == 12345u) dk[0][0] += 1.f;
#endif
    }
#ifndef ABL_ST
    {
      uint8_t* kstg = st + (192 * 2) * ST * 2 + kr * ST * 2;   // this warp's own K / V rows (read only by it, already in registers)
      store16x32(kstg, dk, SCALE, out + ((size_t)a * N + k0 + kr) * ldd + h * 32 + 128, ldd, lane);
      store16x32(kstg + 64 * ST * 2, dv, 1.f, out + ((size_t)a * N + k0 + kr) * ldd + h * 32 + 256, ldd, lane);
    }
#endif
    __syncthreads();
  }
#ifndef ABL_ST
  // dbias partial of this sample chunk, natural layout [h][w][q % 32][key - (32 w - 48)]
  float* pz = partial + (size_t)blockIdx.z * 4 * nwin * 32 * 128;
#pragma unroll
  for (int i = 0; i < 16; ++i) {
#pragma unroll
    for (int e = 0; e < 2; ++e) {
      const int qq = qs + qoff + 8 * i + 2 * t + e;
      if (qq < 0 || qq >= N) continue;
      const int w = qq >> 5;
#pragma unroll
      for (int r = 0; r < 2; ++r) {
        const int j = k0 + kr + g + 8 * r - (32 * w - 48);
        if (j >= 0 && j < 128) pz[((size_t)(h * nwin + w) * 32 + (qq & 31)) * 128 + j] = dba[i][2 * r + e];
      }
    }
  }
}

// ------------------------------------------------------------------------------------------------------------------ pair bias
// bias2[h][w][i][j] = (LN(z[w, i, j, :]) * gamma) . Wb[h]  log2(e), -inf where the key 32 w - 48 + j is outside [0, N) or masked.
// z: [nwin, 32, 128, 16] bf16 (zbf16 = 1) or fp32; a thread per (w, i, j) row.
extern "C" __global__ __launch_bounds__(256) void h_pair_bias(const void* __restrict__ z, const float* __restrict__ gamma, const float* __restrict__ wb,
                                                               const uint8_t* __restrict__ mask, float* __restrict__ out, int N, int rows, float eps,
                                                               int zbf16) {
  pdl_wait();
  const int nwin = N / 32;
  for (int r = blockIdx.x * blockDim.x + threadIdx.x; r < rows; r += gridDim.x * blockDim.x) {
    float x[16];
    if (zbf16) {
      const uint4* p = reinterpret_cast<const uint4*>(reinterpret_cast<const bf16*>(z) + (size_t)r * 16);
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        const uint4 u = p[i];
        const uint32_t w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          x[8 * i + 2 * e] = __uint_as_float(w[e] << 16);
          x[8 * i + 2 * e + 1] = __uint_as_float(w[e] & 0xffff0000u);
        }
      }
    } else {
      const float4* p = reinterpret_cast<const float4*>(reinterpret_cast<const float*>(z) + (size_t)r * 16);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const float4 u = p[i];
        x[4 * i] = u.x; x[4 * i + 1] = u.y; x[4 * i + 2] = u.z; x[4 * i + 3] = u.w;
      }
    }
    float mean = 0.f;
#pragma unroll
    for (int c = 0; c < 16; ++c) mean += x[c];
    mean *= (1.f / 16.f);
    float var = 0.f;
#pragma unroll
    for (int c = 0; c < 16; ++c) { const float d = x[c] - mean; var += d * d; }
    const float rstd = rsqrtf(var * (1.f / 16.f) + eps);
    const int j = r & 127, w = (r >> 12) % nwin;  // r = ((w * 32 + i) * 128 + j)
    const int key = 32 * w - 48 + j;
    const bool ok = key >= 0 && key < N && (mask == nullptr || mask[key]);
#pragma unroll
    for (int c = 0; c < 16; ++c) x[c] = (x[c] - mean) * rstd * gamma[c];
#pragma unroll
    for (int h = 0; h < 4; ++h) {
      float s = 0.f;
#pragma unroll
      for (int c = 0; c < 16; ++c) s = fmaf(x[c], wb[h * 16 + c], s);
      out[(size_t)h * rows + r] = ok ? s * LOG2E : -INFINITY;
    }
  }
}

// bias backward: z [rows, 16] bf16 / fp32 (zbf16), dbias [4, rows] fp32 -> dz (z's dtype) and per-block partials part[blockIdx.x][h * 16 + c] =
// sum_r dbias[h, r] LN(z[r])[c] (local_bias_fin then gives dWb = G gamma and dgamma = sum_h G Wb). wb [4, 16], gamma [16] fp32.
extern "C" __global__ __launch_bounds__(256) void h_pair_bias_bwd(const void* __restrict__ z, const float* __restrict__ wb, const float* __restrict__ gamma,
                                                                   const float* __restrict__ dbias, void* __restrict__ dz, float* __restrict__ part,
                                                                   int rows, float eps, int zbf16) {
  pdl_wait();
  __shared__ float sw[64], red[8][64];
  if (threadIdx.x < 64) sw[threadIdx.x] = wb[threadIdx.x] * gamma[threadIdx.x & 15];
  __syncthreads();
  float G[64];
#pragma unroll
  for (int i = 0; i < 64; ++i) G[i] = 0.f;
  for (int r = blockIdx.x * blockDim.x + threadIdx.x; r < rows; r += gridDim.x * blockDim.x) {
    float x[16];
    if (zbf16) {
      const uint4* p = reinterpret_cast<const uint4*>(reinterpret_cast<const bf16*>(z) + (size_t)r * 16);
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        const uint4 u = p[i];
        const uint32_t w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
        for (int e = 0; e < 4; ++e) { x[8 * i + 2 * e] = __uint_as_float(w[e] << 16); x[8 * i + 2 * e + 1] = __uint_as_float(w[e] & 0xffff0000u); }
      }
    } else {
      const float4* p = reinterpret_cast<const float4*>(reinterpret_cast<const float*>(z) + (size_t)r * 16);
#pragma unroll
      for (int i = 0; i < 4; ++i) { const float4 u = p[i]; x[4 * i] = u.x; x[4 * i + 1] = u.y; x[4 * i + 2] = u.z; x[4 * i + 3] = u.w; }
    }
    float mean = 0.f;
#pragma unroll
    for (int c = 0; c < 16; ++c) mean += x[c];
    mean *= 0.0625f;
    float var = 0.f;
#pragma unroll
    for (int c = 0; c < 16; ++c) { x[c] -= mean; var += x[c] * x[c]; }
    const float rstd = 1.f / sqrtf(var * 0.0625f + eps);
#pragma unroll
    for (int c = 0; c < 16; ++c) x[c] *= rstd;
    float db[4], dzh[16];
#pragma unroll
    for (int h = 0; h < 4; ++h) db[h] = dbias[(size_t)h * rows + r];
#pragma unroll
    for (int c = 0; c < 16; ++c) {
      dzh[c] = db[0] * sw[c] + db[1] * sw[16 + c] + db[2] * sw[32 + c] + db[3] * sw[48 + c];
#pragma unroll
      for (int h = 0; h < 4; ++h) G[h * 16 + c] += db[h] * x[c];
    }
    float m1 = 0.f, m2 = 0.f;
#pragma unroll
    for (int c = 0; c < 16; ++c) { m1 += dzh[c]; m2 += dzh[c] * x[c]; }
    m1 *= 0.0625f; m2 *= 0.0625f;
    float o[16];
#pragma unroll
    for (int c = 0; c < 16; ++c) o[c] = rstd * (dzh[c] - m1 - x[c] * m2);
    if (zbf16) {
      uint32_t w[8];
#pragma unroll
      for (int i = 0; i < 8; ++i) { uint32_t pr; asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(pr) : "f"(o[2 * i]), "f"(o[2 * i + 1])); w[i] = pr; }
      uint4* p = reinterpret_cast<uint4*>(reinterpret_cast<bf16*>(dz) + (size_t)r * 16);
      p[0] = make_uint4(w[0], w[1], w[2], w[3]);
      p[1] = make_uint4(w[4], w[5], w[6], w[7]);
    } else {
      float4* p = reinterpret_cast<float4*>(reinterpret_cast<float*>(dz) + (size_t)r * 16);
#pragma unroll
      for (int i = 0; i < 4; ++i) p[i] = make_float4(o[4 * i], o[4 * i + 1], o[4 * i + 2], o[4 * i + 3]);
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
#endif
}
