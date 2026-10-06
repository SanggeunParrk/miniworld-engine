// oproj_bwd_tf32.cu — the fp32 SWA atom block's out-projection backward on sm_100a, TF32 tensor cores (the fp32 counterpart of
// oproj_bwd.cu; the outputs of the Triton _swa_oproj_bwd_fp32_kernel, every one fp32):
//   q1 = q + gate_a att,  att = gated Wo^T,  gated = sigmoid(g) o:
//   datt = dq1 gate_a;  dgated = datt Wo;  so = sigmoid(g);  dO = dgated so;  dG = dgated o so (1 - so);  gated = so o
//   D[n, h, s] = sum_{d in head h} dO o  (the attention backward's softmax delta);   d gate_a += sum_aug dq1 att
// datt / gated / dO / dG leave rounded to TF32 (cvt.rna): datt is this kernel's MMA operand, dO the attention backward's, dG the qkvg
// backward's, and all four are operands of TF32 weight gradients (cuBLAS dWo / dWg, which round them the same way), so the rounding
// is the one the MMAs would apply; D is formed from the rounded dO the attention backward multiplies with (as the bf16 path forms it
// from its bf16 dO). dq1, att, g, o are read as fp32.
// Transposed (lane = channel): dgated^T [128 C_in][128 rows] = Wo^T datt^T, M = N = K = 128, kind::tf32. A = Wo^T, resident in shared
// memory (rounded on the host: 64 KB as four K-major 128-B-swizzled [128][32] slices); B = the datt tile [128 rows][128 C_out] warpgroup 0
// writes (K-major, four slices, double-buffered).
//   warpgroup 0 (phase A): one C_out channel per thread, the tile rows in turn: coalesced row loads of dq1 / att (a warpgroup reads one
//     512-B row per step), datt into the tile (32 lanes = one 128-B swizzled row: conflict-free) and to global, the d gate_a augment sums
//     in registers (AT = 8: rows 8 sp + at) or per row (other AT).
//   warpgroup 1 (phase C, one tile behind): one C_in channel per thread (its TMEM lane), dgated^T 32 rows at a time, g / o row loads,
//     gated / dO / dG row stores, D as a warp sum (warp q = channels 32 q .. = head q) by a 32-lane reduce-scatter.
// No input staging and no output staging: every row access is a coalesced 512-B warpgroup access, so the smem holds only the operands.
// Tiles of SP augments x AT atoms <= 128 rows (SP = min(A, 16), AT = 128 / SP), persistent CTAs.
// Shared memory: Wo^T 64 KB | datt tiles 2 x 64 KB | barriers  (192.3 KB).   TMEM: dgated^T[2] at 0 / 128 (256 columns).
// Registers: <= 128 / thread (launch bound 512, launched with 384 threads).
// SPDX-License-Identifier: Apache-2.0
#include "sm100.cuh"
#include <type_traits>
using namespace s100;

// ------------------------------------------------------------------ kind::tf32 (as augmented_attention/cuda/sm100/sm100.cuh, measured on B200)
namespace tf {
__host__ __device__ constexpr uint32_t idesc(int M, int N, int a_mn = 0, int b_mn = 0) {
  return (1u << 4) | (2u << 7) | (2u << 10) | ((uint32_t)a_mn << 15) | ((uint32_t)b_mn << 16) | ((uint32_t)(N >> 3) << 17) |
         ((uint32_t)(M >> 4) << 24);
}
DEVI void mma_ss(uint32_t d, uint64_t a, uint64_t b, uint32_t id, uint32_t acc) {
  asm volatile("{ .reg .pred p; setp.ne.b32 p, %4, 0; tcgen05.mma.cta_group::1.kind::tf32 [%0], %1, %2, %3, p; }"
               :: "r"(d), "l"(a), "l"(b), "r"(id), "r"(acc) : "memory");
}
DEVI uint32_t rna(float x) { uint32_t r; asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(x)); return r & 0xffffe000u; }
DEVI float rnaf(float x) { return __uint_as_float(rna(x)); }
}  // namespace tf

constexpr int C = 128, H = 4, MODW = 6 * C;
constexpr int SL = 128 * 128, TILE = 4 * SL;                               // a [128][32] fp32 slice (16 KB); a [128][128] tile
constexpr int O_W = 0, O_B = TILE, O_BAR = O_B + 2 * TILE, SMEM_BYTES = O_BAR + 256;
static_assert(SMEM_BYTES <= 232448, "shared memory");
constexpr uint32_t I_DG = tf::idesc(128, 128);

struct Bars {
  uint64_t wfull, bfull[2], bfree[2], dfull[2], dfree[2];
  uint32_t tmem;
};
DEVI void red1(float* p, float v) { asm volatile("red.global.add.f32 [%0], %1;" :: "l"(p), "f"(v) : "memory"); }
DEVI float pick8(const float (&a)[8], int k) {                           // a[k] for a runtime k without indexing registers
  float v = a[0];
#pragma unroll
  for (int e = 1; e < 8; ++e) v = k == e ? a[e] : v;
  return v;
}
DEVI float ldf(const float* p) { return __ldg(p); }
DEVI float sigm(float x) { return 1.f / (1.f + __expf(-x)); }
// v[j] held by every lane -> lane l returns the sum over the warp of v[l]
DEVI float rscatter32(float (&v)[32], int lane) {
#pragma unroll
  for (int i = 0; i < 16; ++i) { const bool up = lane & 16; const float k = up ? v[i + 16] : v[i], s = up ? v[i] : v[i + 16]; v[i] = k + __shfl_xor_sync(~0u, s, 16); }
#pragma unroll
  for (int i = 0; i < 8; ++i) { const bool up = lane & 8; const float k = up ? v[i + 8] : v[i], s = up ? v[i] : v[i + 8]; v[i] = k + __shfl_xor_sync(~0u, s, 8); }
#pragma unroll
  for (int i = 0; i < 4; ++i) { const bool up = lane & 4; const float k = up ? v[i + 4] : v[i], s = up ? v[i] : v[i + 4]; v[i] = k + __shfl_xor_sync(~0u, s, 4); }
#pragma unroll
  for (int i = 0; i < 2; ++i) { const bool up = lane & 2; const float k = up ? v[i + 2] : v[i], s = up ? v[i] : v[i + 2]; v[i] = k + __shfl_xor_sync(~0u, s, 2); }
  { const bool up = lane & 1; const float k = up ? v[1] : v[0], s = up ? v[0] : v[1]; v[0] = k + __shfl_xor_sync(~0u, s, 1); }
  return v[0];
}

extern "C" __global__ void __launch_bounds__(512, 1)
swa_oproj_bwd_tf32_sm100(const __grid_constant__ CUtensorMap mwot, const float* __restrict__ DQ1, const float* __restrict__ ATT,
                         const float* __restrict__ G, const float* __restrict__ O, const float* __restrict__ MOD,
                         float* __restrict__ DATT, float* __restrict__ GATED, float* __restrict__ DO, float* __restrict__ DG,
                         float* __restrict__ DV, float* __restrict__ DMOD, int S, int A, int Bn, int SP, int AT, int nab, int nag, int ntile) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  Bars& B = *reinterpret_cast<Bars*>(sm + O_BAR);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int R = SP * AT;
  const int ntT = (int)blockIdx.x < ntile ? (ntile - (int)blockIdx.x + (int)gridDim.x - 1) / (int)gridDim.x : 0;
  auto coords = [&](int T, int& b, int& a0, int& s0) {
    const int t = (int)blockIdx.x + T * (int)gridDim.x;
    const int ab = t % nab, r = t / nab, ag = r % nag;
    b = r / nag; a0 = ag * SP; s0 = ab * AT;
  };

  if (tid == 0) {
    mbar_init(&B.wfull, 1);
    for (int i = 0; i < 2; ++i) { mbar_init(&B.bfull[i], 1); mbar_init(&B.bfree[i], 1); mbar_init(&B.dfull[i], 1); mbar_init(&B.dfree[i], 4); }
    fence_barrier_init();
  }
  if (warp == 2) { tmem_alloc(smem_u32(&B.tmem), 256); tmem_relinquish(); }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = B.tmem;

  if (warp == 0) {
    // ------------------------------------------------------------------------------------------------ Wo^T -> shared memory, once
    if (lane == 0 && ntT > 0) {
      mbar_expect_tx(&B.wfull, TILE);
      for (int k = 0; k < 4; ++k) tma_load_2d(su + O_W + k * SL, &mwot, &B.wfull, 32 * k, 0);
    }
  } else if (warp == 1) {
    // ------------------------------------------------------------------------------------------------ MMA issuer: dgated^T = Wo^T datt^T
    if (ntT > 0) mbar_wait(&B.wfull, 0);
    for (int T = 0; T < ntT; ++T) {
      const int x = T & 1;
      mbar_wait(&B.bfull[x], (T >> 1) & 1);                               // datt of tile T is in buffer x
      if (T >= 2) mbar_wait(&B.dfree[x], ((T >> 1) - 1) & 1);              // warpgroup 1 has read dgated^T of tile T - 2
      tc_fence_after();
      if (elect_one()) {
#pragma unroll
        for (int k = 0; k < 16; ++k)                                       // K = 128 C_out: 4 slices x 4 steps of 8 (32 B)
          tf::mma_ss(tmem + x * 128, desc_k128(su + O_W + (k >> 2) * SL) + (uint64_t)((k & 3) * 2),
                     desc_k128(su + O_B + x * TILE + (k >> 2) * SL) + (uint64_t)((k & 3) * 2), I_DG, k > 0 ? 1u : 0u);
        tc_commit(&B.bfree[x]);
        tc_commit(&B.dfull[x]);
      }
      __syncwarp();
    }
  } else if (warp >= 4 && warp < 8) {
    // ------------------------------------------------------------------------------------------------ phase A: datt, d gate_a (channel c)
    const int c = (warp & 3) * 32 + lane;
    const uint32_t cofs = (uint32_t)(c >> 5) * SL + (uint32_t)(c & 3) * 4, cq = (uint32_t)((c & 31) >> 2);
    auto tile = [&](int T, auto at8c) {
      constexpr bool AT8 = decltype(at8c)::value;
      int b, a0, s0; coords(T, b, a0, s0);
      const int x = T & 1;
      if (T >= 2) mbar_wait(&B.bfree[x], ((T >> 1) - 1) & 1);              // the MMA of tile T - 2 has read this buffer
      const uint32_t tb = su + O_B + x * TILE + cofs;
      if constexpr (AT8) {
        float ga[8], gs[8];
#pragma unroll
        for (int k = 0; k < 8; ++k) { ga[k] = s0 + k < S ? ldf(MOD + ((size_t)b * S + s0 + k) * MODW + 2 * C + c) : 0.f; gs[k] = 0.f; }
#pragma unroll 1
        for (int sp = 0; sp < 16; ++sp) {                                  // rows 8 sp + at: augment a0 + sp, atom s0 + at
          const bool aok = a0 + sp < A;
          const size_t g0 = (((size_t)(a0 + sp) * Bn + b) * S + s0) * C + c;
          float d1[8], at_[8];
#pragma unroll
          for (int k = 0; k < 8; ++k) {
            const bool ok = aok && s0 + k < S;
            d1[k] = ok ? ldf(DQ1 + g0 + (size_t)k * C) : 0.f; at_[k] = ok ? ldf(ATT + g0 + (size_t)k * C) : 0.f;
          }
#pragma unroll
          for (int k = 0; k < 8; ++k) {
            const int j = 8 * sp + k;
            const uint32_t r = tf::rna(d1[k] * ga[k]);
            sts32(tb + (uint32_t)j * 128u + ((cq ^ (uint32_t)k) << 4), r);
            if (aok && s0 + k < S) DATT[g0 + (size_t)k * C] = __uint_as_float(r);
            gs[k] += d1[k] * at_[k];                                       // zero on invalid rows
          }
        }
        fence_proxy_async();
        named_bar_sync(1, 128);
        if (c == 0) mbar_arrive(&B.bfull[x]);
        const int rot = (int)blockIdx.x & 7;                               // CTAs start their flushes at different atoms
#pragma unroll
        for (int kk = 0; kk < 8; ++kk) {
          const int k = (kk + rot) & 7;
          if (s0 + k < S) red1(DMOD + ((size_t)b * S + s0 + k) * MODW + 2 * C + c, pick8(gs, k));
        }
      } else {
#pragma unroll 4
        for (int j = 0; j < R; ++j) {
          const int sp = j / AT, at = j - sp * AT;
          const bool ok = a0 + sp < A && s0 + at < S;
          float d1 = 0.f, at_ = 0.f, ga = 0.f;
          const size_t gr = (((size_t)(a0 + sp) * Bn + b) * S + s0 + at) * C + c;
          if (ok) { d1 = ldf(DQ1 + gr); at_ = ldf(ATT + gr); ga = ldf(MOD + ((size_t)b * S + s0 + at) * MODW + 2 * C + c); }
          const uint32_t r = tf::rna(d1 * ga);
          sts32(tb + (uint32_t)j * 128u + ((cq ^ (uint32_t)(j & 7)) << 4), r);
          if (ok) { DATT[gr] = __uint_as_float(r); red1(DMOD + ((size_t)b * S + s0 + at) * MODW + 2 * C + c, d1 * at_); }
        }
        fence_proxy_async();
        named_bar_sync(1, 128);
        if (c == 0) mbar_arrive(&B.bfull[x]);
      }
    };
    for (int T = 0; T < ntT; ++T) {
      if (AT == 8) tile(T, std::true_type{}); else tile(T, std::false_type{});
    }
  } else if (warp >= 8) {
    // ------------------------------------------------------------------------------------------------ phase C: gated, dO, dG, D (C_in channel c)
    const int qw = warp & 3, c = qw * 32 + lane;
    for (int T = 0; T < ntT; ++T) {
      int b, a0, s0; coords(T, b, a0, s0);
      const int x = T & 1;
      mbar_wait(&B.dfull[x], (T >> 1) & 1);
      tc_fence_after();
      const uint32_t tacc = tmem + ((uint32_t)(qw * 32) << 16) + (uint32_t)(x * 128);
#pragma unroll 1
      for (int r0 = 0; r0 < R; r0 += 32) {                                 // 32 rows (TMEM columns) at a time
        uint32_t dv[32];
        tmem_ld32(tacc + (uint32_t)r0, dv);
        float v[32];
#pragma unroll
        for (int q8 = 0; q8 < 4; ++q8) {                                  // 8 rows' loads in flight, then their math
          float gg[8], oo[8];
          uint32_t grow[8];                                                // global row (n S + s; N S < 2^32), ~0 when invalid
#pragma unroll
          for (int e = 0; e < 8; ++e) {
            const int j = r0 + 8 * q8 + e, sp = j / AT, at = j - sp * AT;
            const bool ok = j < R && a0 + sp < A && s0 + at < S;
            grow[e] = ok ? (uint32_t)(((a0 + sp) * Bn + b) * S + s0 + at) : ~0u;
            gg[e] = ok ? ldf(G + (size_t)grow[e] * C + c) : 0.f; oo[e] = ok ? ldf(O + (size_t)grow[e] * C + c) : 0.f;
          }
          if (q8 == 0) tmem_wait_ld();
#pragma unroll
          for (int e = 0; e < 8; ++e) {
            const float dg = __uint_as_float(dv[8 * q8 + e]), so = sigm(gg[e]), oe = oo[e];
            const float dob = tf::rnaf(dg * so);
            const bool ok = grow[e] != ~0u;
            if (ok) {
              const size_t gr = (size_t)grow[e] * C + c;
              DO[gr] = dob; DG[gr] = tf::rnaf(dg * oe * so * (1.f - so)); GATED[gr] = tf::rnaf(so * oe);
            }
            v[8 * q8 + e] = ok ? dob * oe : 0.f;
          }
        }
        const float dsum = rscatter32(v, lane);                            // row r0 + lane, head qw
        const int j = r0 + lane, sp = j / AT, at = j - sp * AT;
        if (j < R && a0 + sp < A && s0 + at < S) DV[(((size_t)(a0 + sp) * Bn + b) * H + qw) * S + s0 + at] = dsum;
      }
      tc_fence_before();
      __syncwarp();
      if (lane == 0) mbar_arrive(&B.dfree[x]);
    }
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 2) { tc_fence_after(); tmem_dealloc(tmem, 256); }
}
