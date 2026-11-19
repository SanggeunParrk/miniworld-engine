// tf8_gemm.cu -- the e4m3 GEMMs of the opt-in B200 (sm_100a) FP8 Transition (route fp8_sm100a.py; docs/b200/fp8.md): one persistent, warp-specialised
// tcgen05 kernel (320 threads, one CTA per SM: warp 0 TMA producer, warp 1 MMA issuer + TMEM owner, warps 2-9 epilogue), C[M, N] = A[M, K] B[N, K]^T
// with K-major 128-B-swizzled operands. -DE4=1: e4m3 operands on kind::f8f6f4 (K = 128 elements per 128-B block, 4 MMAs of K = 32), fp32 accumulation
// in TMEM; -DE4=0: bf16 on kind::f16 (K = 64 per block, 4 MMAs of K = 16) with the same tile, ring and epilogue (the microbenchmark's control).
//   MODE 0 SWIGLU8  h8 = e4m3(silu(acc0 sc[1]) * acc1 sc[2])  (acc0 = A Wa^T, acc1 = A Wb^T, two accumulators); max |h8 value| -> *peak (guard)
//   MODE 1 RESID8   out(bf16) = res + acc sc[3]               (the squeeze: A = h8, B = Ws8)
//   MODE 2 PLAIN    out(bf16) = acc sc[1]                     (microbenchmark)
// The maps are 2-D, 128-B swizzle, box (64 bf16-typed elements = 128 B) x rows: e4m3 tensors are described as bf16 views (TMA moves bytes), so the K
// coordinate of block k is 64 k in both builds.  Epilogue stores go straight from registers (one row per thread, 32-column groups).
// SPDX-License-Identifier: Apache-2.0
#include "sm100.cuh"
using namespace s100;

#ifndef E4
#define E4 1
#endif
#ifndef MODE
#define MODE 0
#endif
#ifndef BN
#define BN 128
#endif
#ifndef ABL
#define ABL 0   // timing ablations only (wrong output): 1 no silu, 2 no stores, 3 no epilogue math and no stores
#endif
constexpr int NACC = MODE == 0 ? 2 : 1;
constexpr int NEPI = 8;
constexpr int A_BYTES = 128 * 128, B_BYTES = BN * 128, STAGE = A_BYTES + NACC * B_BYTES;
constexpr int BUDGET = 232448 - 1024;
constexpr int NST_FIT = BUDGET / STAGE;
constexpr int NST = NST_FIT > 6 ? 6 : NST_FIT;
constexpr int O_BAR = NST * STAGE;
constexpr int SMEM_BYTES = O_BAR + 1024;
constexpr int TCOLS = NACC * BN;
static_assert(2 * TCOLS <= 512 && BN % 32 == 0 && BN <= 256 && NST >= 2, "tile");
static_assert(STAGE % 1024 == 0, "128-B swizzle atoms");

#if E4
constexpr uint32_t IDESC = (1u << 4) | ((uint32_t)(BN >> 3) << 17) | ((uint32_t)(128 >> 4) << 24);   // D f32, A / B e4m3 (format 0), K-major
DEVI void umma(uint32_t d, uint64_t a, uint64_t b, uint32_t acc) {
  asm volatile("{ .reg .pred p; setp.ne.b32 p, %4, 0; tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, p; }"
               :: "r"(d), "l"(a), "l"(b), "r"(IDESC), "r"(acc) : "memory");
}
#else
constexpr uint32_t IDESC = idesc_bf16(128, BN, 0, 0);
DEVI void umma(uint32_t d, uint64_t a, uint64_t b, uint32_t acc) { umma_ss(d, a, b, IDESC, acc); }
#endif

struct Bars {
  uint64_t full[NST], empty[NST], accf[2], acce[2];
  uint32_t tmem;
};

// 4 consecutive e4m3 (a, b, c, d in memory order) packed in a word; saturating (never the NaN byte), round to nearest even
DEVI uint32_t pack_e4m3x4(float a, float b, float c, float d) {
  uint32_t lo, hi;
  asm("{ .reg .b16 r; cvt.rn.satfinite.e4m3x2.f32 r, %1, %2; cvt.u32.u16 %0, r; }" : "=r"(lo) : "f"(b), "f"(a));
  asm("{ .reg .b16 r; cvt.rn.satfinite.e4m3x2.f32 r, %1, %2; cvt.u32.u16 %0, r; }" : "=r"(hi) : "f"(d), "f"(c));
  return lo | (hi << 16);
}
DEVI float silu_t(float a) { const float t = tanhf_approx(0.5f * a); return (0.5f * a) * (1.f + t); }   // silu(a) = (a / 2)(1 + tanh(a / 2)): one MUFU op

extern "C" __global__ void __launch_bounds__(64 + 32 * NEPI, 1)
tf8_gemm_sm100(const __grid_constant__ CUtensorMap mA, const __grid_constant__ CUtensorMap mB0, const __grid_constant__ CUtensorMap mB1,
               void* __restrict__ out, const void* __restrict__ res, const float* __restrict__ sc, float* __restrict__ peak, int M, int N, int K,
               int ldo, int ldres) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  Bars& B = *reinterpret_cast<Bars*>(sm + O_BAR);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  if (threadIdx.x == 0) {
    for (int s = 0; s < NST; ++s) { mbar_init(&B.full[s], 1); mbar_init(&B.empty[s], 1); }
    for (int s = 0; s < 2; ++s) { mbar_init(&B.accf[s], 1); mbar_init(&B.acce[s], NEPI); }
    fence_barrier_init();
    prefetch_map(&mA); prefetch_map(&mB0); if (NACC > 1) prefetch_map(&mB1);
  }
  if (warp == 1) { tmem_alloc(smem_u32(&B.tmem), 512); tmem_relinquish(); }
  tc_fence_before();
  __syncthreads();
  pdl_wait();
  pdl_launch();
  tc_fence_after();
  const uint32_t tmem = B.tmem;
  constexpr int BKE = E4 ? 128 : 64;
  const int NT = N / BN, MT = (M + 127) / 128, tiles = MT * NT, nk = K / BKE;
  const int cta = blockIdx.x, ncta = gridDim.x;

  if (warp == 0) {
    if (lane == 0) {
      int it = 0;
      for (int t = cta; t < tiles; t += ncta) {
        const int m0 = (t / NT) * 128, n0 = (t % NT) * BN;
        for (int k = 0; k < nk; ++k, ++it) {
          const int s = it % NST;
          mbar_wait(&B.empty[s], ((it / NST) & 1) ^ 1);
          mbar_expect_tx(&B.full[s], A_BYTES + NACC * B_BYTES);
          const uint32_t base = su + s * STAGE;
          tma_load_2d(base, &mA, &B.full[s], k * 64, m0);
          tma_load_2d(base + A_BYTES, &mB0, &B.full[s], k * 64, n0);
          if (NACC > 1) tma_load_2d(base + A_BYTES + B_BYTES, &mB1, &B.full[s], k * 64, n0);
        }
      }
    }
  } else if (warp == 1) {
    if (lane == 0) {
      int it = 0, tc = 0;
      for (int t = cta; t < tiles; t += ncta, ++tc) {
        const int buf = tc & 1;
        mbar_wait(&B.acce[buf], ((tc >> 1) & 1) ^ 1);
        tc_fence_after();
        const uint32_t dbase = tmem + buf * TCOLS;
        for (int k = 0; k < nk; ++k, ++it) {
          const int s = it % NST;
          mbar_wait(&B.full[s], (it / NST) & 1);
          tc_fence_after();
          const uint32_t base = su + s * STAGE;
          const uint64_t da = desc_k128(base);
#pragma unroll
          for (int j = 0; j < NACC; ++j) {
            const uint64_t db = desc_k128(base + A_BYTES + j * B_BYTES);
#pragma unroll
            for (int kk = 0; kk < 4; ++kk) umma(dbase + j * BN, da + (uint64_t)(kk * 2), db + (uint64_t)(kk * 2), (k > 0 || kk > 0) ? 1u : 0u);
          }
          tc_commit(&B.empty[s]);
        }
        tc_commit(&B.accf[buf]);
      }
    }
  } else {
    const int ew = warp - 2, q4 = warp & 3, half = ew >> 2;
    float pk = 0.f;
    int tc = 0;
    for (int t = cta; t < tiles; t += ncta, ++tc) {
      const int buf = tc & 1;
      const int m0 = (t / NT) * 128, n0 = (t % NT) * BN;
      const int row = m0 + q4 * 32 + lane;
      mbar_wait(&B.accf[buf], (tc >> 1) & 1);
      tc_fence_after();
      const uint32_t tb = tmem + ((uint32_t)(q4 * 32) << 16) + buf * TCOLS;
      for (int g = half; g < BN / 32; g += 2) {
        const int col = g * 32;
        uint32_t acc[NACC][32];
#pragma unroll
        for (int j = 0; j < NACC; ++j) tmem_ld32(tb + j * BN + col, acc[j]);
        tmem_wait_ld();
        if (row < M) {
#if MODE == 0
          const float ca = sc[1], cb = sc[2];
          float h[32];
#pragma unroll
          for (int e = 0; e < 32; ++e) {
#if ABL == 1 || ABL == 3
            h[e] = __uint_as_float(acc[0][e]) * __uint_as_float(acc[1][e]);
#else
            h[e] = silu_t(__uint_as_float(acc[0][e]) * ca) * (__uint_as_float(acc[1][e]) * cb);
#endif
            pk = fmaxf(pk, fabsf(h[e]));
          }
#if ABL == 2 || ABL == 3
          if (h[0] == 12345.f)
#endif
          {
          uint32_t w[8];
#pragma unroll
          for (int q = 0; q < 8; ++q) w[q] = pack_e4m3x4(h[4 * q], h[4 * q + 1], h[4 * q + 2], h[4 * q + 3]);
          uint8_t* p = reinterpret_cast<uint8_t*>(out) + (size_t)row * ldo + n0 + col;
          stg128(p, make_uint4(w[0], w[1], w[2], w[3]));
          stg128(p + 16, make_uint4(w[4], w[5], w[6], w[7]));
          }
#else
          const float co = MODE == 1 ? sc[3] : sc[1];
          float o[32];
#if MODE == 1
          {
            const __nv_bfloat16* rp = reinterpret_cast<const __nv_bfloat16*>(res) + (size_t)row * ldres + n0 + col;
#pragma unroll
            for (int q = 0; q < 4; ++q) {
              const uint4 w = ldg128(rp + 8 * q);
              const uint32_t ww[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
              for (int e = 0; e < 4; ++e) { o[8 * q + 2 * e] = bf16lo(ww[e]); o[8 * q + 2 * e + 1] = bf16hi(ww[e]); }
            }
          }
#pragma unroll
          for (int e = 0; e < 32; ++e) o[e] += __uint_as_float(acc[0][e]) * co;
#else
#pragma unroll
          for (int e = 0; e < 32; ++e) o[e] = __uint_as_float(acc[0][e]) * co;
#endif
          __nv_bfloat16* op = reinterpret_cast<__nv_bfloat16*>(out) + (size_t)row * ldo + n0 + col;
#pragma unroll
          for (int q = 0; q < 4; ++q)
            stg128(op + 8 * q, make_uint4(pack_bf16(o[8 * q], o[8 * q + 1]), pack_bf16(o[8 * q + 2], o[8 * q + 3]),
                                           pack_bf16(o[8 * q + 4], o[8 * q + 5]), pack_bf16(o[8 * q + 6], o[8 * q + 7])));
#endif
        }
      }
      tc_fence_before();
      __syncwarp();
      if (lane == 0) mbar_arrive(&B.acce[buf]);
    }
#if MODE == 0
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) pk = fmaxf(pk, __shfl_xor_sync(0xffffffffu, pk, o));
    if (lane == 0 && pk > 0.f) atomicMax(reinterpret_cast<int*>(peak), __float_as_int(pk));
#endif
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 1) { tc_fence_after(); tmem_dealloc(tmem, 512); }
}

extern "C" __device__ int transition_smem_bytes = SMEM_BYTES;
