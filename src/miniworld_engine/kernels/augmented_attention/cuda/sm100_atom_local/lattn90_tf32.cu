// lattn90_tf32.cu — AF3 windowed atom attention, fp32 activations with TF32 tensor-core products, on sm_90a (mma.sync m16n8k8).
//
// Same geometry as lattn90.cu: CTA = 64 queries (2 windows) x 1 head x a chunk of samples, warp = 16 queries, K / V = the 160-key union,
// the pair bias (log2 domain, key mask folded in as -inf) in shared memory once per CTA. q, k, v arrive rounded to tf32 (the row kernels
// round them on the way out); P is rounded to nearest tf32 before the PV product; softmax statistics and O stay fp32.
//   S = q k^T / sqrt(32) + bias        O = softmax(S) v        (fp32 O [A N, 128], row LSE [A, 4, N] in log2 units)
// The k slots of the PV product are permuted (slot t <-> key 8 nt + 2 t, slot t + 4 <-> key 8 nt + 2 t + 1) so that the S accumulator
// is the A operand in place.
// SPDX-License-Identifier: Apache-2.0
#include <cuda.h>
#include <stdint.h>
#define DEVI __device__ __forceinline__

namespace {
constexpr int ST = 36;     // fp32 row stride of Q / K / V tiles (conflict-free fragment loads)
constexpr int BSQ = 136;   // fp32 row stride of the bias tile
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
DEVI void mma(float* d, const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
DEVI uint32_t tf(float x) { uint32_t r; asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(x)); return r; }
DEVI float ex2(float x) { float y; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
DEVI float quad_max(float v) {
  v = fmaxf(v, __shfl_xor_sync(~0u, v, 1));
  return fmaxf(v, __shfl_xor_sync(~0u, v, 2));
}
DEVI float quad_sum(float v) {
  v += __shfl_xor_sync(~0u, v, 1);
  return v + __shfl_xor_sync(~0u, v, 2);
}
DEVI void load_rows(uint32_t dst, const float* x, int a, int h, int r0, int n, int N, int tid) {
  for (int i = tid; i < n * 8; i += 128) {
    const int r = i >> 3, c = i & 7, gr = r0 + r;
    const bool ok = gr >= 0 && gr < N;
    cp16(dst + (r * ST + c * 4) * 4, x + ((size_t)a * N + (ok ? gr : 0)) * 128 + h * 32 + c * 4, ok);
  }
}
}  // namespace

// dynamic smem: bias [64][BSQ] f32, Q [64][ST], K [160][ST], V [160][ST] f32
constexpr int F_SMEM = 64 * BSQ * 4 + (64 + 160 + 160) * ST * 4;

extern "C" __global__ __launch_bounds__(128) void h_attn_fwd_tf32(const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v,
                                                                   const float* __restrict__ bias, float* __restrict__ out, float* __restrict__ lse,
                                                                   int N, int A, int spc) {
  pdl_wait();
  extern __shared__ __align__(16) uint8_t sm[];
  float* sb = reinterpret_cast<float*>(sm);
  float* sQ = sb + 64 * BSQ;
  float* sK = sQ + 64 * ST;
  float* sV = sK + 160 * ST;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const int q0 = 64 * blockIdx.x, h = blockIdx.y, a0 = blockIdx.z * spc, a1 = min(A, a0 + spc), nwin = N / 32, w0 = q0 / 32;
  const int win = warp >> 1;
  for (int i = tid; i < 64 * 32; i += 128) {
    const int r = i >> 5, c = i & 31;
    cp16f(su(sb + r * BSQ + c * 4), bias + ((size_t)(h * nwin + w0) * 32 + r) * 128 + c * 4);
  }
  for (int a = a0; a < a1; ++a) {
    load_rows(su(sQ), q, a, h, q0, 64, N, tid);
    load_rows(su(sK), k, a, h, q0 - 48, 160, N, tid);
    load_rows(su(sV), v, a, h, q0 - 48, 160, N, tid);
    cp_commit();
    cp_wait<0>();
    __syncthreads();
    uint32_t aq[4][4];
#pragma unroll
    for (int kt = 0; kt < 4; ++kt) {
      const float* r0 = sQ + (16 * warp + g) * ST + 8 * kt + t;
      aq[kt][0] = __float_as_uint(r0[0]);
      aq[kt][1] = __float_as_uint(r0[8 * ST]);
      aq[kt][2] = __float_as_uint(r0[4]);
      aq[kt][3] = __float_as_uint(r0[8 * ST + 4]);
    }
    float sc[16][4];
#pragma unroll
    for (int nt = 0; nt < 16; ++nt) {
      sc[nt][0] = sc[nt][1] = sc[nt][2] = sc[nt][3] = 0.f;
      const float* kr = sK + (32 * win + 8 * nt + g) * ST + t;
#pragma unroll
      for (int kt = 0; kt < 4; ++kt) mma(sc[nt], aq[kt], __float_as_uint(kr[8 * kt]), __float_as_uint(kr[8 * kt + 4]));
    }
    float m0 = -INFINITY, m1 = -INFINITY;
    const float* br = sb + (16 * warp + g) * BSQ + 2 * t;
#pragma unroll
    for (int nt = 0; nt < 16; ++nt) {
      const float2 b0 = *reinterpret_cast<const float2*>(br + 8 * nt), b1 = *reinterpret_cast<const float2*>(br + 8 * BSQ + 8 * nt);
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
    for (int nt = 0; nt < 16; ++nt) {
      const float p0 = ex2(sc[nt][0] - n0), p1 = ex2(sc[nt][1] - n0), p2 = ex2(sc[nt][2] - n1), p3 = ex2(sc[nt][3] - n1);
      l0 += p0 + p1;
      l1 += p2 + p3;
      const uint32_t ap[4] = {tf(p0), tf(p2), tf(p1), tf(p3)};
      const float* vr = sV + (32 * win + 8 * nt + 2 * t) * ST + g;
#pragma unroll
      for (int d = 0; d < 4; ++d) mma(o[d], ap, __float_as_uint(vr[8 * d]), __float_as_uint(vr[ST + 8 * d]));
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
    if (t == 0) {
      lse[((size_t)a * 4 + h) * N + r0] = l0 > 0.f ? n0 + __log2f(l0) : 1e30f;
      lse[((size_t)a * 4 + h) * N + r1] = l1 > 0.f ? n1 + __log2f(l1) : 1e30f;
    }
    __syncthreads();
  }
}

DEVI void load_vec(float* dst, const float* v, int a, int h, int r0, int n, int N, int tid) {
  for (int i = tid; i < n / 4; i += 128) {
    const int gr = r0 + 4 * i;
    const bool ok = gr >= 0 && gr < N;
    cp16(su(dst + 4 * i), v + ((size_t)a * 4 + h) * N + (ok ? gr : 0), ok);
  }
}

// ----------------------------------------------------------------------------------------------------------------- dQ
// dynamic smem: bias [64][BSQ] f32, Q, dO [64][ST], K, V [160][ST] f32, LSE, D [64] f32
constexpr int Q_SMEM = 64 * BSQ * 4 + (64 * 2 + 160 * 2) * ST * 4 + 2 * 64 * 4;

extern "C" __global__ __launch_bounds__(128) void h_attn_dq_tf32(const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v,
                                                                  const float* __restrict__ dout, const float* __restrict__ bias,
                                                                  const float* __restrict__ lse, const float* __restrict__ dd, float* __restrict__ out,
                                                                  int ldd, int N, int A, int spc) {
  pdl_wait();
  extern __shared__ __align__(16) uint8_t sm[];
  float* sb = reinterpret_cast<float*>(sm);
  float* sQ = sb + 64 * BSQ;
  float* sD = sQ + 64 * ST;
  float* sK = sD + 64 * ST;
  float* sV = sK + 160 * ST;
  float* sl = sV + 160 * ST;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const int q0 = 64 * blockIdx.x, h = blockIdx.y, a0 = blockIdx.z * spc, a1 = min(A, a0 + spc), nwin = N / 32, w0 = q0 / 32;
  const int win = warp >> 1;
  for (int i = tid; i < 64 * 32; i += 128) {
    const int r = i >> 5, c = i & 31;
    cp16f(su(sb + r * BSQ + c * 4), bias + ((size_t)(h * nwin + w0) * 32 + r) * 128 + c * 4);
  }
  for (int a = a0; a < a1; ++a) {
    load_rows(su(sQ), q, a, h, q0, 64, N, tid);
    load_rows(su(sD), dout, a, h, q0, 64, N, tid);
    load_rows(su(sK), k, a, h, q0 - 48, 160, N, tid);
    load_rows(su(sV), v, a, h, q0 - 48, 160, N, tid);
    load_vec(sl, lse, a, h, q0, 64, N, tid);
    load_vec(sl + 64, dd, a, h, q0, 64, N, tid);
    cp_commit();
    cp_wait<0>();
    __syncthreads();
    uint32_t aq[4][4], ad[4][4];
#pragma unroll
    for (int kt = 0; kt < 4; ++kt) {
      const float* r0 = sQ + (16 * warp + g) * ST + 8 * kt + t;
      const float* r1 = sD + (16 * warp + g) * ST + 8 * kt + t;
      aq[kt][0] = __float_as_uint(r0[0]); aq[kt][1] = __float_as_uint(r0[8 * ST]); aq[kt][2] = __float_as_uint(r0[4]); aq[kt][3] = __float_as_uint(r0[8 * ST + 4]);
      ad[kt][0] = __float_as_uint(r1[0]); ad[kt][1] = __float_as_uint(r1[8 * ST]); ad[kt][2] = __float_as_uint(r1[4]); ad[kt][3] = __float_as_uint(r1[8 * ST + 4]);
    }
    const float lse0 = sl[16 * warp + g], lse1 = sl[16 * warp + g + 8], d0 = sl[64 + 16 * warp + g], d1 = sl[64 + 16 * warp + g + 8];
    float dq[4][4];
#pragma unroll
    for (int d = 0; d < 4; ++d) dq[d][0] = dq[d][1] = dq[d][2] = dq[d][3] = 0.f;
    const float* br = sb + (16 * warp + g) * BSQ + 2 * t;
#pragma unroll
    for (int half = 0; half < 2; ++half) {
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const int nt = 8 * half + i;
        float sc[4] = {0.f, 0.f, 0.f, 0.f}, dp[4] = {0.f, 0.f, 0.f, 0.f};
        const float* kr = sK + (32 * win + 8 * nt + g) * ST + t;
        const float* vr = sV + (32 * win + 8 * nt + g) * ST + t;
#pragma unroll
        for (int kt = 0; kt < 4; ++kt) {
          mma(sc, aq[kt], __float_as_uint(kr[8 * kt]), __float_as_uint(kr[8 * kt + 4]));
          mma(dp, ad[kt], __float_as_uint(vr[8 * kt]), __float_as_uint(vr[8 * kt + 4]));
        }
        const float2 b0 = *reinterpret_cast<const float2*>(br + 8 * nt), b1 = *reinterpret_cast<const float2*>(br + 8 * BSQ + 8 * nt);
        const float p0 = ex2(fmaf(sc[0], C1, b0.x) - lse0), p1 = ex2(fmaf(sc[1], C1, b0.y) - lse0);
        const float p2 = ex2(fmaf(sc[2], C1, b1.x) - lse1), p3 = ex2(fmaf(sc[3], C1, b1.y) - lse1);
        const uint32_t ap[4] = {tf(p0 * (dp[0] - d0)), tf(p2 * (dp[2] - d1)), tf(p1 * (dp[1] - d0)), tf(p3 * (dp[3] - d1))};
        const float* kc = sK + (32 * win + 8 * nt + 2 * t) * ST + g;
#pragma unroll
        for (int d = 0; d < 4; ++d) mma(dq[d], ap, __float_as_uint(kc[8 * d]), __float_as_uint(kc[ST + 8 * d]));
      }
    }
    const int r0 = q0 + 16 * warp + g, r1 = r0 + 8;
#pragma unroll
    for (int d = 0; d < 4; ++d) {
      *reinterpret_cast<float2*>(out + ((size_t)a * N + r0) * ldd + h * 32 + 8 * d + 2 * t) = make_float2(dq[d][0] * SCALE, dq[d][1] * SCALE);
      *reinterpret_cast<float2*>(out + ((size_t)a * N + r1) * ldd + h * 32 + 8 * d + 2 * t) = make_float2(dq[d][2] * SCALE, dq[d][3] * SCALE);
    }
    __syncthreads();
  }
}

// ----------------------------------------------------------------------------------------------------------------- dK, dV, dbias
constexpr int BSK = 193;
constexpr int BSW = 132;   // per-warp band: 16 keys x 128 queries (row stride 132: float2 reads)
constexpr int K_SMEM = 4 * 16 * BSW * 4 + (192 * 2 + 64 * 2) * ST * 4 + 2 * 192 * 4;

extern "C" __global__ __launch_bounds__(128) void h_attn_dkv_tf32(const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v,
                                                                   const float* __restrict__ dout, const float* __restrict__ bias,
                                                                   const float* __restrict__ lse, const float* __restrict__ dd, float* __restrict__ out,
                                                                   float* __restrict__ partial, int ldd, int N, int A, int spc) {
  pdl_wait();
  extern __shared__ __align__(16) uint8_t sm[];
  float* sb = reinterpret_cast<float*>(sm);
  float* sQ = sb + 4 * 16 * BSW;
  float* sD = sQ + 192 * ST;
  float* sK = sD + 192 * ST;
  float* sV = sK + 64 * ST;
  float* sl = sV + 64 * ST;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const int m = blockIdx.x, k0 = 64 * m, qs = k0 - 64, h = blockIdx.y, a0 = blockIdx.z * spc, a1 = min(A, a0 + spc), nwin = N / 32;
  const int kr = 16 * warp, qoff = 32 * (((warp - 3) >> 1) + 2);
  // band of the bias, per warp: sb[warp][key row 0..15][query column 0..127], query = qs + qoff(warp) + column
  for (int i = tid; i < 4 * 16 * BSW; i += 128) sb[i] = -INFINITY;
  __syncthreads();
  for (int i = tid; i < 64 * 128; i += 128) {          // every (key kk of the CTA, query column qc of its warp's band)
    const int kk = i >> 7, qc = i & 127, w = kk >> 4;
    const int qq = qs + 32 * (((w - 3) >> 1) + 2) + qc;  // atom index of the query
    if (qq < 0 || qq >= N) continue;
    const int win = qq >> 5, j = k0 + kk - (32 * win - 48);
    if (j >= 0 && j < 128) sb[(w * 16 + (kk & 15)) * BSW + qc] = bias[((size_t)(h * nwin + win) * 32 + (qq & 31)) * 128 + j];
  }
  float dba[16][4];
#pragma unroll
  for (int i = 0; i < 16; ++i) dba[i][0] = dba[i][1] = dba[i][2] = dba[i][3] = 0.f;
  for (int a = a0; a < a1; ++a) {
    __syncthreads();
    load_rows(su(sQ), q, a, h, qs, 192, N, tid);
    load_rows(su(sD), dout, a, h, qs, 192, N, tid);
    load_rows(su(sK), k, a, h, k0, 64, N, tid);
    load_rows(su(sV), v, a, h, k0, 64, N, tid);
    load_vec(sl, lse, a, h, qs, 192, N, tid);
    load_vec(sl + 192, dd, a, h, qs, 192, N, tid);
    cp_commit();
    cp_wait<0>();
    __syncthreads();
    uint32_t ak[4][4], av[4][4];
#pragma unroll
    for (int kt = 0; kt < 4; ++kt) {
      const float* r0 = sK + (kr + g) * ST + 8 * kt + t;
      const float* r1 = sV + (kr + g) * ST + 8 * kt + t;
      ak[kt][0] = __float_as_uint(r0[0]); ak[kt][1] = __float_as_uint(r0[8 * ST]); ak[kt][2] = __float_as_uint(r0[4]); ak[kt][3] = __float_as_uint(r0[8 * ST + 4]);
      av[kt][0] = __float_as_uint(r1[0]); av[kt][1] = __float_as_uint(r1[8 * ST]); av[kt][2] = __float_as_uint(r1[4]); av[kt][3] = __float_as_uint(r1[8 * ST + 4]);
    }
    float dk[4][4], dv[4][4];
#pragma unroll
    for (int n = 0; n < 4; ++n) dk[n][0] = dk[n][1] = dk[n][2] = dk[n][3] = dv[n][0] = dv[n][1] = dv[n][2] = dv[n][3] = 0.f;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      const int qr = qoff + 8 * i;
      float sc[4] = {0.f, 0.f, 0.f, 0.f}, dp[4] = {0.f, 0.f, 0.f, 0.f};
      const float* qp = sQ + (qr + g) * ST + t;
      const float* dpp = sD + (qr + g) * ST + t;
#pragma unroll
      for (int kt = 0; kt < 4; ++kt) {
        mma(sc, ak[kt], __float_as_uint(qp[8 * kt]), __float_as_uint(qp[8 * kt + 4]));
        mma(dp, av[kt], __float_as_uint(dpp[8 * kt]), __float_as_uint(dpp[8 * kt + 4]));
      }
      const int c0 = qr + 2 * t;
      const float2 lq = *reinterpret_cast<const float2*>(sl + c0), dq2 = *reinterpret_cast<const float2*>(sl + 192 + c0);
      const float* bk0 = sb + (warp * 16 + g) * BSW + 8 * i + 2 * t;
      const float2 b0 = *reinterpret_cast<const float2*>(bk0), b1 = *reinterpret_cast<const float2*>(bk0 + 8 * BSW);
      const float p0 = ex2(fmaf(sc[0], C1, b0.x) - lq.x), p1 = ex2(fmaf(sc[1], C1, b0.y) - lq.y);
      const float p2 = ex2(fmaf(sc[2], C1, b1.x) - lq.x), p3 = ex2(fmaf(sc[3], C1, b1.y) - lq.y);
      const float s0 = p0 * (dp[0] - dq2.x), s1 = p1 * (dp[1] - dq2.y), s2 = p2 * (dp[2] - dq2.x), s3 = p3 * (dp[3] - dq2.y);
      dba[i][0] += s0; dba[i][1] += s1; dba[i][2] += s2; dba[i][3] += s3;
      const uint32_t ap[4] = {tf(p0), tf(p2), tf(p1), tf(p3)};
      const uint32_t as[4] = {tf(s0), tf(s2), tf(s1), tf(s3)};
      const float* dr = sD + (qr + 2 * t) * ST + g;
      const float* qc = sQ + (qr + 2 * t) * ST + g;
#pragma unroll
      for (int d = 0; d < 4; ++d) {
        mma(dv[d], ap, __float_as_uint(dr[8 * d]), __float_as_uint(dr[ST + 8 * d]));
        mma(dk[d], as, __float_as_uint(qc[8 * d]), __float_as_uint(qc[ST + 8 * d]));
      }
    }
    const int r0 = k0 + kr + g, r1 = r0 + 8;
#pragma unroll
    for (int d = 0; d < 4; ++d) {
      const size_t i0 = ((size_t)a * N + r0) * ldd + h * 32 + 8 * d + 2 * t, i1 = ((size_t)a * N + r1) * ldd + h * 32 + 8 * d + 2 * t;
      *reinterpret_cast<float2*>(out + i0 + 128) = make_float2(dk[d][0] * SCALE, dk[d][1] * SCALE);
      *reinterpret_cast<float2*>(out + i1 + 128) = make_float2(dk[d][2] * SCALE, dk[d][3] * SCALE);
      *reinterpret_cast<float2*>(out + i0 + 256) = make_float2(dv[d][0], dv[d][1]);
      *reinterpret_cast<float2*>(out + i1 + 256) = make_float2(dv[d][2], dv[d][3]);
    }
  }
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
