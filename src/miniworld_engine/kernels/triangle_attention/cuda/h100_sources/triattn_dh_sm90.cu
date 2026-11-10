// triattn_dh_sm90.cu -- triangle-attention core for head widths 32 / 64 (Chai-1's 4 x 64 trunk heads; the wgmma core in triattn_sm90.cu
// takes 32 only), inference, sm_90a, plain mma.sync (m16n8k16 bf16, fp32 accumulation), FlashAttention-2 schedule:
//
//   u[b,n,s,h,:] = sigmoid(g[b,n,s,h,:] (+ gb[h,:])) * softmax_k( scale q[b,n,s,h,:] . k[b,n,k,h,:] + bias[b,h,s,k] ) @ v[b,n,k,h,:]
//
// q | k | v | g are column blocks of one projection row (b, n, s) of width 4 HD (HD = H D): q at h D, k at HD + h D, v at 2 HD + h D,
// g at 3 HD + h D. bias [B, H, S, S] bf16 in natural units (masked keys -inf), shared by the N pair rows. u [B N S, HD] bf16.
//
// CTA = 64 queries of one (b, n, h); 4 warps x 16 queries; key tiles of 64 through a two-stage cp.async ring (K, V and the bias tile).
// Per key tile a warp computes S = Q K^T (16 x 64), adds the bias, keeps a running maximum (log2 units) and sum, and accumulates
// O += P V with P rounded to bf16 in registers (the S accumulator re-read as the A operand).
//
// Training (the width-generic H100 path's core for 64-channel heads, h100_sm90_generic.py): the same forward with TRAIN also writes
// O = softmax(..) v (bf16 [B N S, HD]) and the base-2 log-sum-exp LSE [B, N, H, S] (x = log2e (scale q.k + bias), lse = log2 sum 2^x;
// +inf for a row without an unmasked key), and two backward kernels recompute P = 2^(x - lse) from it (FlashAttention-2 split, the
// same mma.sync fragments; Delta = rowsum(dO o O) comes from the glue's gate backward):
//   bwd_kv  CTA = 64 keys of one (b, n, h), 4 warps x 16 keys, query tiles of 64 streamed:  S^T = K Q^T + bias^T, dP^T = V dO^T,
//           dS^T = P^T o (dP^T - Delta),  dV += P^T dO,  dK += dS^T Q  (dK scaled at the end)
//   bwd_q   CTA = 64 queries of one (b, h) x a slab of the pair rows n, 4 warps x 16 queries, key tiles of 64:  dQ += dS K per row n
//           (scaled, stored per row), and dbias[slab][b, h, q, k] += dS: every element of a slab is owned by one thread of one CTA
//           (plain read-modify-write in a fixed order, no atomics); the slabs are summed in a fixed order by the caller (deterministic).
// dS enters the dQ / dK products as a bf16 hi + lo pair (two MMAs: a near-fp32 operand, as the fp32-softmax references compute them).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <stdint.h>

namespace {
typedef __nv_bfloat16 bf16;
#define DEVI __device__ __forceinline__

constexpr int BM = 64, BN = 64, NW = 4, THREADS = 32 * NW;
constexpr float LOG2E = 1.4426950408889634f;

DEVI uint32_t su(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
DEVI void cp16(uint32_t d, const void* s) { asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(d), "l"(s) : "memory"); }
DEVI void cp_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
template <int N> DEVI void cp_wait() { asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory"); }
DEVI void ldsm4(uint32_t* r, uint32_t a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
DEVI void ldsm4t(uint32_t* r, uint32_t a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
DEVI void mma(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
DEVI uint32_t pk(float lo, float hi) {
  __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&v);
}
// the bf16 remainder of (a, b) after its bf16 pair `hi` (dS as hi + lo: two MMAs, near-fp32 operand)
DEVI uint32_t pk_lo(float a, float b, uint32_t hi) {
  const __nv_bfloat162 h = *reinterpret_cast<const __nv_bfloat162*>(&hi);
  return pk(a - __low2float(h), b - __high2float(h));
}
DEVI float ex2(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
DEVI float quad_max(float v) {
  v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 1));
  return fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 2));
}
DEVI float quad_sum(float v) {
  v += __shfl_xor_sync(0xffffffffu, v, 1);
  return v + __shfl_xor_sync(0xffffffffu, v, 2);
}

// shared tiles: rows of D (+8 pad) bf16 for Q / K / V, 64 (+8 pad) for the bias -- conflict-free ldmatrix
template <int D> struct Smem {
  static constexpr int LD = D + 8, LB = BN + 8;
  bf16 q[BM * LD];
  bf16 k[2][BN * LD];
  bf16 v[2][BN * LD];
  bf16 b[2][BM * LB];
};

template <int D, bool TRAIN = false>
__global__ void __launch_bounds__(THREADS) triattn_dh_fwd(const bf16* __restrict__ P, const bf16* __restrict__ BIAS, bf16* __restrict__ U,
    const float* __restrict__ GB, int S, int H, float scale, bf16* __restrict__ O = nullptr, float* __restrict__ LSE = nullptr) {
  constexpr int LD = Smem<D>::LD, LB = Smem<D>::LB, KD = D / 16, ND = D / 8;
  extern __shared__ __align__(16) unsigned char smem_raw[];
  Smem<D>& sm = *reinterpret_cast<Smem<D>*>(smem_raw);
  const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32, g = lane / 4, t = lane % 4;
  const int q0 = blockIdx.x * BM, h = blockIdx.y;
  const long bn = blockIdx.z;                      // b * N + n (N == S)
  const int b = (int)(bn / S);
  const int HD = H * D, ld = 4 * HD;
  const bf16* rows = P + bn * S * ld;              // the S projection rows of pair row (b, n)
  const bf16* bias = BIAS + ((long)b * H + h) * S * S;

  // Q tile and the first K | V | bias stage
  constexpr int QC = BM * D / 8;                   // 16-byte chunks of a Q / K / V tile
  for (int c = tid; c < QC; c += THREADS) {
    const int r = c / (D / 8), col = (c % (D / 8)) * 8;
    cp16(su(&sm.q[r * LD + col]), rows + (long)(q0 + r) * ld + h * D + col);
  }
  auto stage = [&](int j, int s) {
    const int k0 = j * BN;
    for (int c = tid; c < QC; c += THREADS) {
      const int r = c / (D / 8), col = (c % (D / 8)) * 8;
      cp16(su(&sm.k[s][r * LD + col]), rows + (long)(k0 + r) * ld + HD + h * D + col);
      cp16(su(&sm.v[s][r * LD + col]), rows + (long)(k0 + r) * ld + 2 * HD + h * D + col);
    }
    for (int c = tid; c < BM * BN / 8; c += THREADS) {
      const int r = c / (BN / 8), col = (c % (BN / 8)) * 8;
      cp16(su(&sm.b[s][r * LB + col]), bias + (long)(q0 + r) * S + k0 + col);
    }
  };
  stage(0, 0);
  cp_commit();

  uint32_t qa[KD][4];
  float o[ND][4];
#pragma unroll
  for (int n = 0; n < ND; ++n) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.f;
  float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;
  const float sc = scale * LOG2E;
  const int ntiles = S / BN;
  for (int j = 0; j < ntiles; ++j) {
    const int s = j & 1;
    if (j + 1 < ntiles) stage(j + 1, s ^ 1);
    cp_commit();
    cp_wait<1>();
    __syncthreads();
    if (j == 0) {
#pragma unroll
      for (int kk = 0; kk < KD; ++kk) ldsm4(qa[kk], su(&sm.q[(warp * 16 + lane % 16) * LD + kk * 16 + (lane / 16) * 8]));
    }
    // S = Q K^T: 8 n-tiles of 8 keys
    float sacc[BN / 8][4];
#pragma unroll
    for (int n = 0; n < BN / 8; ++n) sacc[n][0] = sacc[n][1] = sacc[n][2] = sacc[n][3] = 0.f;
#pragma unroll
    for (int kk = 0; kk < KD; ++kk) {
#pragma unroll
      for (int n = 0; n < BN / 8; n += 2) {
        uint32_t kb[4];
        ldsm4(kb, su(&sm.k[s][(n * 8 + (lane % 8) + (lane / 16) * 8) * LD + kk * 16 + ((lane / 8) % 2) * 8]));
        mma(sacc[n], qa[kk], kb[0], kb[1]);
        mma(sacc[n + 1], qa[kk], kb[2], kb[3]);
      }
    }
    // log2-unit logits with the bias; running maximum and sum per row (g, g + 8)
    const int r0 = warp * 16 + g;
    float mx0 = m0, mx1 = m1;
#pragma unroll
    for (int n = 0; n < BN / 8; ++n) {
      const int col = n * 8 + 2 * t;
      const __nv_bfloat162 b0 = *reinterpret_cast<const __nv_bfloat162*>(&sm.b[s][r0 * LB + col]);
      const __nv_bfloat162 b1 = *reinterpret_cast<const __nv_bfloat162*>(&sm.b[s][(r0 + 8) * LB + col]);
      sacc[n][0] = sacc[n][0] * sc + __low2float(b0) * LOG2E;
      sacc[n][1] = sacc[n][1] * sc + __high2float(b0) * LOG2E;
      sacc[n][2] = sacc[n][2] * sc + __low2float(b1) * LOG2E;
      sacc[n][3] = sacc[n][3] * sc + __high2float(b1) * LOG2E;
      mx0 = fmaxf(mx0, fmaxf(sacc[n][0], sacc[n][1]));
      mx1 = fmaxf(mx1, fmaxf(sacc[n][2], sacc[n][3]));
    }
    mx0 = quad_max(mx0);
    mx1 = quad_max(mx1);
    // a row whose keys are all masked so far keeps a finite offset (its p are 0)
    const float off0 = mx0 == -INFINITY ? 0.f : mx0, off1 = mx1 == -INFINITY ? 0.f : mx1;
    const float a0 = ex2((m0 == -INFINITY ? off0 : m0) - off0), a1 = ex2((m1 == -INFINITY ? off1 : m1) - off1);
    m0 = mx0;
    m1 = mx1;
    l0 *= a0;
    l1 *= a1;
#pragma unroll
    for (int n = 0; n < ND; ++n) { o[n][0] *= a0; o[n][1] *= a0; o[n][2] *= a1; o[n][3] *= a1; }
    uint32_t pa[BN / 16][4];
#pragma unroll
    for (int n = 0; n < BN / 8; ++n) {
      const float p0 = ex2(sacc[n][0] - off0), p1 = ex2(sacc[n][1] - off0), p2 = ex2(sacc[n][2] - off1), p3 = ex2(sacc[n][3] - off1);
      l0 += p0 + p1;
      l1 += p2 + p3;
      pa[n / 2][(n % 2) * 2] = pk(p0, p1);
      pa[n / 2][(n % 2) * 2 + 1] = pk(p2, p3);
    }
    // O += P V: k = 64 keys (4 steps of 16), n = D dims
#pragma unroll
    for (int kk = 0; kk < BN / 16; ++kk) {
#pragma unroll
      for (int n = 0; n < ND; n += 2) {
        uint32_t vb[4];
        ldsm4t(vb, su(&sm.v[s][(kk * 16 + (lane % 8) + ((lane / 8) % 2) * 8) * LD + n * 8 + (lane / 16) * 8]));
        mma(o[n], pa[kk], vb[0], vb[1]);
        mma(o[n + 1], pa[kk], vb[2], vb[3]);
      }
    }
    __syncthreads();
  }
  l0 = quad_sum(l0);
  l1 = quad_sum(l1);
  const float i0 = l0 > 0.f ? 1.f / l0 : 0.f, i1 = l1 > 0.f ? 1.f / l1 : 0.f;
  if constexpr (TRAIN) {
    if (t == 0) {
      float* lse = LSE + (bn * H + h) * S + q0 + warp * 16 + g;
      lse[0] = l0 > 0.f ? m0 + __log2f(l0) : INFINITY;
      lse[8] = l1 > 0.f ? m1 + __log2f(l1) : INFINITY;
    }
  }
  // u = sigmoid(g (+ gb)) o, bf16 into U [B N S, HD]
  const long row0 = bn * S + q0 + warp * 16 + g;
#pragma unroll
  for (int n = 0; n < ND; ++n) {
    const int col = h * D + n * 8 + 2 * t;
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      const long row = row0 + half * 8;
      const __nv_bfloat162 gv = *reinterpret_cast<const __nv_bfloat162*>(P + row * ld + 3 * HD + col);
      float ga = __low2float(gv), gb = __high2float(gv);
      if (GB != nullptr) { ga += GB[col]; gb += GB[col + 1]; }
      const float inv = half ? i1 : i0;
      const float v0 = o[n][2 * half] * inv / (1.f + __expf(-ga)), v1 = o[n][2 * half + 1] * inv / (1.f + __expf(-gb));
      *reinterpret_cast<uint32_t*>(U + row * HD + col) = pk(v0, v1);
      if constexpr (TRAIN) *reinterpret_cast<uint32_t*>(O + row * HD + col) = pk(o[n][2 * half] * inv, o[n][2 * half + 1] * inv);
    }
  }
}

// ---------------------------------------------------------------------------------------------------------------------------------
// backward, key side: dK, dV of 64 keys of one (b, n, h)
template <int D> struct SmemKV {
  static constexpr int LD = D + 8, LB = BN + 8;
  bf16 k[BN * LD];
  bf16 v[BN * LD];
  bf16 q[2][BM * LD];
  bf16 d[2][BM * LD];       // dO
  bf16 b[2][BM * LB];       // bias tile [query][key]
  float lse[2][BM];
  float dl[2][BM];          // Delta
};

template <int D>
__global__ void __launch_bounds__(THREADS) triattn_dh_bwd_kv(const bf16* __restrict__ P, const bf16* __restrict__ BIAS, const bf16* __restrict__ DO,
    const float* __restrict__ LSE, const float* __restrict__ DELTA, bf16* __restrict__ DK, bf16* __restrict__ DV, long ldd, int S, int H,
    float scale) {
  constexpr int LD = SmemKV<D>::LD, LB = SmemKV<D>::LB, KD = D / 16, ND = D / 8;
  extern __shared__ __align__(16) unsigned char smem_raw[];
  SmemKV<D>& sm = *reinterpret_cast<SmemKV<D>*>(smem_raw);
  const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32, g = lane / 4, t = lane % 4;
  const int k0 = blockIdx.x * BN, h = blockIdx.y;
  const long bn = blockIdx.z;
  const int b = (int)(bn / S);
  const int HD = H * D, ld = 4 * HD;
  const bf16* rows = P + bn * S * ld;
  const bf16* drows = DO + bn * S * HD;
  const bf16* bias = BIAS + ((long)b * H + h) * S * S;
  const float* lse = LSE + (bn * H + h) * S;
  const float* delta = DELTA + (bn * H + h) * S;
  constexpr int QC = BM * D / 8;
  for (int c = tid; c < QC; c += THREADS) {
    const int r = c / (D / 8), col = (c % (D / 8)) * 8;
    cp16(su(&sm.k[r * LD + col]), rows + (long)(k0 + r) * ld + HD + h * D + col);
    cp16(su(&sm.v[r * LD + col]), rows + (long)(k0 + r) * ld + 2 * HD + h * D + col);
  }
  auto stage = [&](int j, int s) {
    const int qq = j * BM;
    for (int c = tid; c < QC; c += THREADS) {
      const int r = c / (D / 8), col = (c % (D / 8)) * 8;
      cp16(su(&sm.q[s][r * LD + col]), rows + (long)(qq + r) * ld + h * D + col);
      cp16(su(&sm.d[s][r * LD + col]), drows + (long)(qq + r) * HD + h * D + col);
    }
    for (int c = tid; c < BM * BN / 8; c += THREADS) {
      const int r = c / (BN / 8), col = (c % (BN / 8)) * 8;
      cp16(su(&sm.b[s][r * LB + col]), bias + (long)(qq + r) * S + k0 + col);
    }
    if (tid < BM / 4) cp16(su(&sm.lse[s][tid * 4]), lse + qq + tid * 4);
    else if (tid < BM / 2) cp16(su(&sm.dl[s][(tid - BM / 4) * 4]), delta + qq + (tid - BM / 4) * 4);
  };
  stage(0, 0);
  cp_commit();
  uint32_t ka[KD][4], va[KD][4];
  float dk[ND][4], dv[ND][4];
#pragma unroll
  for (int n = 0; n < ND; ++n)
#pragma unroll
    for (int e = 0; e < 4; ++e) dk[n][e] = dv[n][e] = 0.f;
  const float sc = scale * LOG2E;
  const int ntiles = S / BM;
  const int rk0 = warp * 16 + g;
  for (int j = 0; j < ntiles; ++j) {
    const int s = j & 1;
    if (j + 1 < ntiles) stage(j + 1, s ^ 1);
    cp_commit();
    cp_wait<1>();
    __syncthreads();
    if (j == 0) {
#pragma unroll
      for (int kk = 0; kk < KD; ++kk) {
        ldsm4(ka[kk], su(&sm.k[(warp * 16 + lane % 16) * LD + kk * 16 + (lane / 16) * 8]));
        ldsm4(va[kk], su(&sm.v[(warp * 16 + lane % 16) * LD + kk * 16 + (lane / 16) * 8]));
      }
    }
    // S^T = K Q^T, dP^T = V dO^T: 16 keys x 64 queries per warp
    float st[BM / 8][4], dp[BM / 8][4];
#pragma unroll
    for (int n = 0; n < BM / 8; ++n)
#pragma unroll
      for (int e = 0; e < 4; ++e) st[n][e] = dp[n][e] = 0.f;
#pragma unroll
    for (int kk = 0; kk < KD; ++kk) {
#pragma unroll
      for (int n = 0; n < BM / 8; n += 2) {
        uint32_t qb[4], db[4];
        const int off = (n * 8 + (lane % 8) + (lane / 16) * 8) * LD + kk * 16 + ((lane / 8) % 2) * 8;
        ldsm4(qb, su(&sm.q[s][off]));
        ldsm4(db, su(&sm.d[s][off]));
        mma(st[n], ka[kk], qb[0], qb[1]);
        mma(st[n + 1], ka[kk], qb[2], qb[3]);
        mma(dp[n], va[kk], db[0], db[1]);
        mma(dp[n + 1], va[kk], db[2], db[3]);
      }
    }
    // P^T = 2^(x - lse), dS^T = P^T o (dP^T - Delta); rows = keys rk0 / rk0 + 8, columns = queries
    uint32_t pa[BM / 16][4], sa[BM / 16][4], sl[BM / 16][4];
#pragma unroll
    for (int n = 0; n < BM / 8; ++n) {
      const int qc = n * 8 + 2 * t;
      const float la = sm.lse[s][qc], lb = sm.lse[s][qc + 1], da = sm.dl[s][qc], dd = sm.dl[s][qc + 1];
      const float b00 = __bfloat162float(sm.b[s][qc * LB + rk0]), b01 = __bfloat162float(sm.b[s][(qc + 1) * LB + rk0]);
      const float b10 = __bfloat162float(sm.b[s][qc * LB + rk0 + 8]), b11 = __bfloat162float(sm.b[s][(qc + 1) * LB + rk0 + 8]);
      const float p0 = ex2(st[n][0] * sc + b00 * LOG2E - la), p1 = ex2(st[n][1] * sc + b01 * LOG2E - lb);
      const float p2 = ex2(st[n][2] * sc + b10 * LOG2E - la), p3 = ex2(st[n][3] * sc + b11 * LOG2E - lb);
      pa[n / 2][(n % 2) * 2] = pk(p0, p1);
      pa[n / 2][(n % 2) * 2 + 1] = pk(p2, p3);
      const float s0 = p0 * (dp[n][0] - da), s1 = p1 * (dp[n][1] - dd), s2 = p2 * (dp[n][2] - da), s3 = p3 * (dp[n][3] - dd);
      sa[n / 2][(n % 2) * 2] = pk(s0, s1);
      sa[n / 2][(n % 2) * 2 + 1] = pk(s2, s3);
      sl[n / 2][(n % 2) * 2] = pk_lo(s0, s1, sa[n / 2][(n % 2) * 2]);
      sl[n / 2][(n % 2) * 2 + 1] = pk_lo(s2, s3, sa[n / 2][(n % 2) * 2 + 1]);
    }
    // dV += P^T dO, dK += dS^T Q: k = 64 queries (4 steps of 16), n = D dims
#pragma unroll
    for (int kk = 0; kk < BM / 16; ++kk) {
#pragma unroll
      for (int n = 0; n < ND; n += 2) {
        uint32_t db[4], qb[4];
        const int off = (kk * 16 + (lane % 8) + ((lane / 8) % 2) * 8) * LD + n * 8 + (lane / 16) * 8;
        ldsm4t(db, su(&sm.d[s][off]));
        ldsm4t(qb, su(&sm.q[s][off]));
        mma(dv[n], pa[kk], db[0], db[1]);
        mma(dv[n + 1], pa[kk], db[2], db[3]);
        mma(dk[n], sa[kk], qb[0], qb[1]);
        mma(dk[n + 1], sa[kk], qb[2], qb[3]);
        mma(dk[n], sl[kk], qb[0], qb[1]);
        mma(dk[n + 1], sl[kk], qb[2], qb[3]);
      }
    }
    __syncthreads();
  }
  const long krow = bn * S + k0 + rk0;
#pragma unroll
  for (int n = 0; n < ND; ++n) {
    const int col = h * D + n * 8 + 2 * t;
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      const long row = krow + half * 8;
      *reinterpret_cast<uint32_t*>(DK + row * ldd + col) = pk(dk[n][2 * half] * scale, dk[n][2 * half + 1] * scale);
      *reinterpret_cast<uint32_t*>(DV + row * ldd + col) = pk(dv[n][2 * half], dv[n][2 * half + 1]);
    }
  }
}

// ---------------------------------------------------------------------------------------------------------------------------------
// backward, query side: dQ of 64 queries of one (b, h) for the pair rows of a slab, and the slab's dbias partial
template <int D> struct SmemQ {
  static constexpr int LD = D + 8, LB = BN + 8;
  bf16 q[BM * LD];
  bf16 d[BM * LD];          // dO
  bf16 k[2][BN * LD];
  bf16 v[2][BN * LD];
  bf16 b[2][BM * LB];
};

// ROWS (the launched build): one pair row per slab (NSLAB = S), the row's dS written once as a bf16 plane (DB is then bf16 [S, B, H, S, S]:
// no read-modify-write, a grid of B S (S / 64) H short CTAs) and dQ += dS K on the bf16 dS alone -- the data flow and the operand of the
// Triton backward; the caller sums the planes in fp32. ROWS false: fp32 slab partials read-modified-written row by row, dS as hi + lo.
template <int D, bool ROWS = false>
__global__ void __launch_bounds__(THREADS) triattn_dh_bwd_q(const bf16* __restrict__ P, const bf16* __restrict__ BIAS, const bf16* __restrict__ DO,
    const float* __restrict__ LSE, const float* __restrict__ DELTA, bf16* __restrict__ DQ, float* __restrict__ DB, long ldd, int B, int S, int H,
    int NSLAB, float scale) {
  constexpr int LD = SmemQ<D>::LD, LB = SmemQ<D>::LB, KD = D / 16, ND = D / 8;
  extern __shared__ __align__(16) unsigned char smem_raw[];
  SmemQ<D>& sm = *reinterpret_cast<SmemQ<D>*>(smem_raw);
  const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32, g = lane / 4, t = lane % 4;
  const int q0 = blockIdx.x * BM, h = blockIdx.y;
  const int b = (int)(blockIdx.z / NSLAB), p = (int)(blockIdx.z % NSLAB);
  const int n0 = (int)((long)p * S / NSLAB), n1 = (int)((long)(p + 1) * S / NSLAB);
  const int HD = H * D, ld = 4 * HD;
  const bf16* bias = BIAS + ((long)b * H + h) * S * S;
  float* dbias = DB + (((long)p * B + b) * H + h) * S * S;
  const float sc = scale * LOG2E;
  const int ntiles = S / BN;
  const int r0 = warp * 16 + g;
  constexpr int QC = BM * D / 8;
  for (int n = n0; n < n1; ++n) {
    const long bn = (long)b * S + n;
    const bf16* rows = P + bn * S * ld;
    const bf16* drows = DO + bn * S * HD;
    for (int c = tid; c < QC; c += THREADS) {
      const int r = c / (D / 8), col = (c % (D / 8)) * 8;
      cp16(su(&sm.q[r * LD + col]), rows + (long)(q0 + r) * ld + h * D + col);
      cp16(su(&sm.d[r * LD + col]), drows + (long)(q0 + r) * HD + h * D + col);
    }
    auto stage = [&](int j, int s) {
      const int kk0 = j * BN;
      for (int c = tid; c < QC; c += THREADS) {
        const int r = c / (D / 8), col = (c % (D / 8)) * 8;
        cp16(su(&sm.k[s][r * LD + col]), rows + (long)(kk0 + r) * ld + HD + h * D + col);
        cp16(su(&sm.v[s][r * LD + col]), rows + (long)(kk0 + r) * ld + 2 * HD + h * D + col);
      }
      for (int c = tid; c < BM * BN / 8; c += THREADS) {
        const int r = c / (BN / 8), col = (c % (BN / 8)) * 8;
        cp16(su(&sm.b[s][r * LB + col]), bias + (long)(q0 + r) * S + kk0 + col);
      }
    };
    stage(0, 0);
    cp_commit();
    const float* lse = LSE + (bn * H + h) * S + q0;
    const float* delta = DELTA + (bn * H + h) * S + q0;
    const float la = lse[r0], lb = lse[r0 + 8], da = delta[r0], dd = delta[r0 + 8];
    uint32_t qa[KD][4], doa[KD][4];
    float dq[ND][4];
#pragma unroll
    for (int nn = 0; nn < ND; ++nn)
#pragma unroll
      for (int e = 0; e < 4; ++e) dq[nn][e] = 0.f;
    for (int j = 0; j < ntiles; ++j) {
      const int s = j & 1;
      if (j + 1 < ntiles) stage(j + 1, s ^ 1);
      cp_commit();
      cp_wait<1>();
      __syncthreads();
      if (j == 0) {
#pragma unroll
        for (int kk = 0; kk < KD; ++kk) {
          ldsm4(qa[kk], su(&sm.q[(warp * 16 + lane % 16) * LD + kk * 16 + (lane / 16) * 8]));
          ldsm4(doa[kk], su(&sm.d[(warp * 16 + lane % 16) * LD + kk * 16 + (lane / 16) * 8]));
        }
      }
      // the slab's partial so far (the first row of the slab writes it)
      float* dbt = dbias + (long)(q0 + r0) * S + j * BN + 2 * t;
      float2 old[ROWS ? 1 : BN / 8][2];
#pragma unroll
      for (int nn = 0; nn < (ROWS ? 0 : BN / 8); ++nn) {
        if (n == n0) { old[nn][0] = make_float2(0.f, 0.f); old[nn][1] = make_float2(0.f, 0.f); }
        else {
          old[nn][0] = *reinterpret_cast<const float2*>(dbt + nn * 8);
          old[nn][1] = *reinterpret_cast<const float2*>(dbt + 8L * S + nn * 8);
        }
      }
      float sacc[BN / 8][4], dp[BN / 8][4];
#pragma unroll
      for (int nn = 0; nn < BN / 8; ++nn)
#pragma unroll
        for (int e = 0; e < 4; ++e) sacc[nn][e] = dp[nn][e] = 0.f;
#pragma unroll
      for (int kk = 0; kk < KD; ++kk) {
#pragma unroll
        for (int nn = 0; nn < BN / 8; nn += 2) {
          uint32_t kb[4], vb[4];
          const int off = (nn * 8 + (lane % 8) + (lane / 16) * 8) * LD + kk * 16 + ((lane / 8) % 2) * 8;
          ldsm4(kb, su(&sm.k[s][off]));
          ldsm4(vb, su(&sm.v[s][off]));
          mma(sacc[nn], qa[kk], kb[0], kb[1]);
          mma(sacc[nn + 1], qa[kk], kb[2], kb[3]);
          mma(dp[nn], doa[kk], vb[0], vb[1]);
          mma(dp[nn + 1], doa[kk], vb[2], vb[3]);
        }
      }
      uint32_t sa[BN / 16][4], sl[BN / 16][4];
#pragma unroll
      for (int nn = 0; nn < BN / 8; ++nn) {
        const int col = nn * 8 + 2 * t;
        const __nv_bfloat162 b0 = *reinterpret_cast<const __nv_bfloat162*>(&sm.b[s][r0 * LB + col]);
        const __nv_bfloat162 b1 = *reinterpret_cast<const __nv_bfloat162*>(&sm.b[s][(r0 + 8) * LB + col]);
        const float p0 = ex2(sacc[nn][0] * sc + __low2float(b0) * LOG2E - la), p1 = ex2(sacc[nn][1] * sc + __high2float(b0) * LOG2E - la);
        const float p2 = ex2(sacc[nn][2] * sc + __low2float(b1) * LOG2E - lb), p3 = ex2(sacc[nn][3] * sc + __high2float(b1) * LOG2E - lb);
        const float s0 = p0 * (dp[nn][0] - da), s1 = p1 * (dp[nn][1] - da), s2 = p2 * (dp[nn][2] - dd), s3 = p3 * (dp[nn][3] - dd);
        sa[nn / 2][(nn % 2) * 2] = pk(s0, s1);
        sa[nn / 2][(nn % 2) * 2 + 1] = pk(s2, s3);
        sl[nn / 2][(nn % 2) * 2] = pk_lo(s0, s1, sa[nn / 2][(nn % 2) * 2]);
        sl[nn / 2][(nn % 2) * 2 + 1] = pk_lo(s2, s3, sa[nn / 2][(nn % 2) * 2 + 1]);
        if constexpr (ROWS) {
          bf16* dbh = reinterpret_cast<bf16*>(DB) + ((((long)p * B + b) * H + h) * S + q0 + r0) * S + j * BN + 2 * t;
          *reinterpret_cast<uint32_t*>(dbh + nn * 8) = pk(s0, s1);
          *reinterpret_cast<uint32_t*>(dbh + 8L * S + nn * 8) = pk(s2, s3);
        } else {
          *reinterpret_cast<float2*>(dbt + nn * 8) = make_float2(old[nn][0].x + s0, old[nn][0].y + s1);
          *reinterpret_cast<float2*>(dbt + 8L * S + nn * 8) = make_float2(old[nn][1].x + s2, old[nn][1].y + s3);
        }
      }
      // dQ += dS K: k = 64 keys (4 steps of 16), n = D dims
#pragma unroll
      for (int kk = 0; kk < BN / 16; ++kk) {
#pragma unroll
        for (int nn = 0; nn < ND; nn += 2) {
          uint32_t kb[4];
          ldsm4t(kb, su(&sm.k[s][(kk * 16 + (lane % 8) + ((lane / 8) % 2) * 8) * LD + nn * 8 + (lane / 16) * 8]));
          mma(dq[nn], sa[kk], kb[0], kb[1]);
          mma(dq[nn + 1], sa[kk], kb[2], kb[3]);
          if constexpr (!ROWS) {
            mma(dq[nn], sl[kk], kb[0], kb[1]);
            mma(dq[nn + 1], sl[kk], kb[2], kb[3]);
          }
        }
      }
      __syncthreads();
    }
    const long qrow = bn * S + q0 + r0;
#pragma unroll
    for (int nn = 0; nn < ND; ++nn) {
      const int col = h * D + nn * 8 + 2 * t;
      *reinterpret_cast<uint32_t*>(DQ + qrow * ldd + col) = pk(dq[nn][0] * scale, dq[nn][1] * scale);
      *reinterpret_cast<uint32_t*>(DQ + (qrow + 8) * ldd + col) = pk(dq[nn][2] * scale, dq[nn][3] * scale);
    }
  }
}

template <int D, bool TRAIN = false>
void launch(const at::Tensor& proj, const at::Tensor& bias, at::Tensor& u, const float* gb, int B, int S, int H, double scale,
            bf16* o = nullptr, float* lse = nullptr) {
  auto f = triattn_dh_fwd<D, TRAIN>;
  const int smem = (int)sizeof(Smem<D>);
  cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  dim3 grid(S / BM, H, (unsigned)B * S);
  f<<<grid, THREADS, smem, at::cuda::getCurrentCUDAStream()>>>(reinterpret_cast<const bf16*>(proj.data_ptr()),
      reinterpret_cast<const bf16*>(bias.data_ptr()), reinterpret_cast<bf16*>(u.data_ptr()), gb, S, H, (float)scale, o, lse);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int D>
void launch_bwd(const at::Tensor& proj, const at::Tensor& bias, const at::Tensor& dout, const at::Tensor& lse, const at::Tensor& delta,
                at::Tensor& dall, int64_t dqcol, int64_t dkcol, int64_t dvcol, at::Tensor& dbias, int B, int S, int H, int nslab, double scale, bool rows) {
  auto stream = at::cuda::getCurrentCUDAStream();
  const bf16* P = reinterpret_cast<const bf16*>(proj.data_ptr());
  const bf16* BI = reinterpret_cast<const bf16*>(bias.data_ptr());
  const bf16* DOp = reinterpret_cast<const bf16*>(dout.data_ptr());
  bf16* DA = reinterpret_cast<bf16*>(dall.data_ptr());
  const long ldd = dall.stride(0);
  {
    auto f = triattn_dh_bwd_kv<D>;
    const int smem = (int)sizeof(SmemKV<D>);
    cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    dim3 grid(S / BN, H, (unsigned)B * S);
    f<<<grid, THREADS, smem, stream>>>(P, BI, DOp, lse.data_ptr<float>(), delta.data_ptr<float>(), DA + dkcol, DA + dvcol, ldd, S, H, (float)scale);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  {
    auto f = rows ? triattn_dh_bwd_q<D, true> : triattn_dh_bwd_q<D, false>;
    const int smem = (int)sizeof(SmemQ<D>);
    cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    dim3 grid(S / BM, H, (unsigned)B * nslab);
    f<<<grid, THREADS, smem, stream>>>(P, BI, DOp, lse.data_ptr<float>(), delta.data_ptr<float>(), DA + dqcol, reinterpret_cast<float*>(dbias.data_ptr()), ldd, B, S,
                                       H, nslab, (float)scale);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}

void check_core(const at::Tensor& proj, const at::Tensor& bias, int64_t H, int64_t D) {
  TORCH_CHECK(D == 32 || D == 64, "head width 32 or 64");
  TORCH_CHECK(proj.is_cuda() && proj.scalar_type() == at::kBFloat16 && proj.is_contiguous() && proj.dim() == 2 && proj.size(1) == 4 * H * D,
              "proj: contiguous bf16 [B S S, 4 H D]");
  TORCH_CHECK(bias.scalar_type() == at::kBFloat16 && bias.is_contiguous() && bias.dim() == 4 && bias.size(1) == H && bias.size(2) == bias.size(3),
              "bias: contiguous bf16 [B, H, S, S]");
  TORCH_CHECK(bias.size(2) % 64 == 0 && proj.size(0) == bias.size(0) * bias.size(2) * bias.size(2), "S a multiple of 64, proj rows B S S");
}
}  // namespace

// proj [B S S, 4 H D] bf16 (q | k | v | g), bias [B, H, S, S] bf16 -> u [B S S, H D] bf16. gate_bias: fp32 [H D] or empty.
at::Tensor triattn_dh_forward(at::Tensor proj, at::Tensor bias, at::Tensor gate_bias, int64_t H, int64_t D, double scale) {
  check_core(proj, bias, H, D);
  const int B = (int)bias.size(0), S = (int)bias.size(2);
  TORCH_CHECK(gate_bias.numel() == 0 || (gate_bias.scalar_type() == at::kFloat && gate_bias.is_contiguous() && gate_bias.numel() == H * D),
              "gate_bias: fp32 [H D] or empty");
  const at::cuda::OptionalCUDAGuard guard(proj.device());
  auto u = at::empty({proj.size(0), H * D}, proj.options());
  const float* gb = gate_bias.numel() ? gate_bias.data_ptr<float>() : nullptr;
  if (D == 64) launch<64>(proj, bias, u, gb, B, S, (int)H, scale);
  else launch<32>(proj, bias, u, gb, B, S, (int)H, scale);
  return u;
}

// training forward: -> (o [B S S, H D] bf16, lse [B, S, H, S] fp32 base 2, u [B S S, H D] bf16); the gate is read from proj as it is
std::vector<at::Tensor> triattn_dh_forward_train(at::Tensor proj, at::Tensor bias, int64_t H, int64_t D, double scale) {
  check_core(proj, bias, H, D);
  const int B = (int)bias.size(0), S = (int)bias.size(2);
  const at::cuda::OptionalCUDAGuard guard(proj.device());
  auto u = at::empty({proj.size(0), H * D}, proj.options());
  auto o = at::empty({proj.size(0), H * D}, proj.options());
  auto lse = at::empty({B, S, H, S}, proj.options().dtype(at::kFloat));
  bf16* op = reinterpret_cast<bf16*>(o.data_ptr());
  if (D == 64) launch<64, true>(proj, bias, u, nullptr, B, S, (int)H, scale, op, lse.data_ptr<float>());
  else launch<32, true>(proj, bias, u, nullptr, B, S, (int)H, scale, op, lse.data_ptr<float>());
  return {o, lse, u};
}

// backward: dout [B S S, H D] bf16, lse / delta fp32 [B, S, H, S]; dq / dk / dv are written into columns dqcol / dkcol / dvcol (+ h D) of
// dall [B S S, *] bf16 (row stride dall.stride(0)) -> dbias slabs fp32 [nslab, B, H, S, S] (the caller sums them over the first axis)
at::Tensor triattn_dh_backward(at::Tensor proj, at::Tensor bias, at::Tensor dout, at::Tensor lse, at::Tensor delta, at::Tensor dall, int64_t dqcol,
                               int64_t dkcol, int64_t dvcol, int64_t H, int64_t D, double scale, int64_t nslab, bool rows) {
  check_core(proj, bias, H, D);
  const int B = (int)bias.size(0), S = (int)bias.size(2);
  TORCH_CHECK(dout.scalar_type() == at::kBFloat16 && dout.is_contiguous() && dout.size(0) == proj.size(0) && dout.size(1) == H * D, "dout [B S S, H D]");
  TORCH_CHECK(lse.scalar_type() == at::kFloat && lse.is_contiguous() && lse.numel() == (int64_t)B * S * H * S, "lse fp32 [B, S, H, S]");
  TORCH_CHECK(delta.scalar_type() == at::kFloat && delta.is_contiguous() && delta.numel() == (int64_t)B * S * H * S, "delta fp32 [B, S, H, S]");
  TORCH_CHECK(dall.scalar_type() == at::kBFloat16 && dall.dim() == 2 && dall.stride(1) == 1 && dall.size(0) == proj.size(0) &&
              dall.stride(0) % 8 == 0 && dqcol % 8 == 0 && dkcol % 8 == 0 && dvcol % 8 == 0, "dall bf16 [B S S, *], 16-byte aligned columns");
  TORCH_CHECK(nslab >= 1 && nslab <= S, "1 <= nslab <= S");
  const at::cuda::OptionalCUDAGuard guard(proj.device());
  TORCH_CHECK(!rows || nslab == S, "rows: one slab per pair row");
  auto dbias = at::empty({nslab, B, H, S, S}, proj.options().dtype(rows ? at::kBFloat16 : at::kFloat));
  if (D == 64) launch_bwd<64>(proj, bias, dout, lse, delta, dall, dqcol, dkcol, dvcol, dbias, B, S, (int)H, (int)nslab, scale, rows);
  else launch_bwd<32>(proj, bias, dout, lse, delta, dall, dqcol, dkcol, dvcol, dbias, B, S, (int)H, (int)nslab, scale, rows);
  return dbias;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("triattn_dh_forward", &triattn_dh_forward);
  m.def("triattn_dh_forward_train", &triattn_dh_forward_train);
  m.def("triattn_dh_backward", &triattn_dh_backward);
}
