// bo_logits_sm100.cu -- the exact-fp32 pair-bias LOGITS of the bias-only token DiT's hoist for bf16 pair rows, sm_100a (tcgen05):
//
//   logits[h, r] = sum_c W'[h, c] LN0(x[r])[c]            W' = W diag(gamma) fp32 [16, CP], x [R, CP] bf16, CP = 128 or 256, R a multiple of 128
//
// A bf16 row is an exact operand, so for a row with |mean| <= std / 2 the logit is rs (sum_c W'_c x_c - mean sum_c W'_c): x (bf16, exact) times W' split into
// three bf16 pieces (hi + mid + lo: 24 bits, the fp32 weight) on the bf16 tensor core -- every product exact in fp32, one accumulator per 64-column slab, the
// slabs and pieces added in fp32 (smallest first) in the epilogue -- and the mean term from sum_c W'_c in fp64. The mean-subtraction cancels harmlessly there. A row
// with a larger mean (|mean| > std / 2) takes the LN0-first path on the FMA pipe (the row's exact fp32 LN0 values, fp32 products): the error class of the exact
// FMA GEMM in both cases (tests/gpu/test_bias_only_fp32_logits.py). The row statistics (two exact passes over the row in shared memory) are written as (mean, rstd)
// when asked (the training backward's input).
//
// A work item is a tile of 128 rows (TMA: CP / 64 slabs of [128 rows][64 cols] SW128 = one stage); M 128 x N 48 (3 pieces x 16 heads) x K 16 MMAs, one TMEM
// accumulator per slab, two accumulator buffers. Warps: 0 TMA producer, 1 MMA, 2 TMEM allocator, 4-7 epilogue (one row a thread), 8-11 the row statistics
// (one row a thread, a tile ahead of the epilogue, through a two-slot shared ring).
// SPDX-License-Identifier: Apache-2.0
#include "sm100.cuh"
using namespace s100;

#ifndef CP
#define CP 256
#endif
#if CP == 256   // the 64-register-deep unroll of the four slabs spilled / serialised the statistics warps (220 us against 77 us at L 768)
#define SUNR_PRAGMA _Pragma("unroll 1")
#else
#define SUNR_PRAGMA _Pragma("unroll")
#endif
#ifndef EPSV
#define EPSV 1e-5f
#endif
constexpr int QM = 128, NS = CP / 64, NB = 48, NCOL = NS * NB;
constexpr int TS = QM * 128, STB = NS * TS, BS = NB * 128;     // slab, stage, one B slab (48 rows of 128 B)
constexpr int O_B = 0, O_ST = ((NS * BS + 1023) / 1024) * 1024;
constexpr int NST = (232448 - O_ST - 3072) / STB > 4 ? 4 : (232448 - O_ST - 3072) / STB;
constexpr int O_BAR = O_ST + NST * STB, SMEM_BYTES = O_BAR + 3072;
static_assert(NST >= 2 && SMEM_BYTES <= 232448, "shared memory");
constexpr uint32_t IDESC = idesc_bf16(QM, NB), TCOLS = 2 * NCOL <= 256 ? 256 : 512;
struct Bars { uint64_t full[NST], empty[NST], acc[2], acc_empty[2], sfull[2], sempty[2]; uint32_t tmem; float wsum[16]; float2 st[2][QM]; };

DEVI float sum8(const uint4& v) {
  return ((bf16lo(v.x) + bf16hi(v.x)) + (bf16lo(v.y) + bf16hi(v.y))) + ((bf16lo(v.z) + bf16hi(v.z)) + (bf16lo(v.w) + bf16hi(v.w)));
}
DEVI float sq8(const uint4& v, float m) {
  float d[8] = {bf16lo(v.x) - m, bf16hi(v.x) - m, bf16lo(v.y) - m, bf16hi(v.y) - m, bf16lo(v.z) - m, bf16hi(v.z) - m, bf16lo(v.w) - m, bf16hi(v.w) - m};
  return ((d[0] * d[0] + d[1] * d[1]) + (d[2] * d[2] + d[3] * d[3])) + ((d[4] * d[4] + d[5] * d[5]) + (d[6] * d[6] + d[7] * d[7]));
}
DEVI uint16_t bf16_bits_rn(float x) { uint32_t r; asm("cvt.rn.bf16x2.f32 %0, 0f00000000, %1;" : "=r"(r) : "f"(x)); return (uint16_t)(r & 0xffffu); }

extern "C" __global__ void __launch_bounds__(384, 1)
bo_logits_sm100(const __grid_constant__ CUtensorMap mp, const float* __restrict__ WF, float* __restrict__ LG, float2* __restrict__ PST, int R, int use_pst) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  Bars& B = *reinterpret_cast<Bars*>(sm + O_BAR);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int tiles = R / QM, vb = blockIdx.x, vg = gridDim.x;
  const int my = tiles > vb ? (tiles - vb + vg - 1) / vg : 0;

  if (tid == 0) {
    for (int s = 0; s < NST; ++s) { mbar_init(&B.full[s], 1); mbar_init(&B.empty[s], 4); }
    for (int e = 0; e < 2; ++e) { mbar_init(&B.acc[e], 1); mbar_init(&B.acc_empty[e], 4); mbar_init(&B.sfull[e], 128); mbar_init(&B.sempty[e], 128); }
    fence_barrier_init();
    prefetch_map(&mp);
  }
  if (warp == 2) { tmem_alloc(smem_u32(&B.tmem), TCOLS); tmem_relinquish(); }
  // W' in three bf16 pieces (lo, mid, hi: n = 16 piece + head), K-major SW128 slabs of 48 rows
  for (int i = tid; i < NB * CP; i += blockDim.x) {
    const int n = i / CP, k = i % CP, h = n & 15, p = n >> 4;
    const float w = WF[h * CP + k];
    const uint16_t hi = bf16_bits_rn(w);
    const float r1 = w - __uint_as_float((uint32_t)hi << 16);
    const uint16_t mid = bf16_bits_rn(r1);
    const float r2 = r1 - __uint_as_float((uint32_t)mid << 16);
    const uint16_t lo = bf16_bits_rn(r2);
    const uint16_t val = p == 0 ? lo : (p == 1 ? mid : hi);
    const uint32_t a = O_B + (k / 64) * BS + n * 128 + ((((k % 64) / 8) ^ (n & 7)) << 4) + (k % 8) * 2;
    *reinterpret_cast<uint16_t*>(sm + a) = val;
  }
  if (warp < 12) {   // sum_c W'[h, c] in fp64 (a warp a head, heads 12 .. 15 on the first four warps again)
    for (int h = warp; h < 16; h += 12) {
      double sd = 0.0;
      for (int c = lane; c < CP; c += 32) sd += (double)WF[h * CP + c];
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) sd += __shfl_xor_sync(0xffffffffu, sd, o);
      if (lane == 0) B.wsum[h] = (float)sd;
    }
  }
  fence_proxy_async();
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = B.tmem;

  if (warp == 0) {
    if (lane == 0) {
      for (int li = 0; li < my; ++li) {
        const int s = li % NST;
        if (li >= NST) mbar_wait(&B.empty[s], ((li / NST) - 1) & 1);
        const uint32_t st = su + O_ST + s * STB;
        mbar_expect_tx(&B.full[s], STB);
        const int r0 = (vb + li * vg) * QM;
#pragma unroll
        for (int c = 0; c < NS; ++c) tma_load_2d(st + c * TS, &mp, &B.full[s], c * 64, r0);
      }
    }
  } else if (warp == 1) {
    for (int li = 0; li < my; ++li) {
      const int s = li % NST, buf = li & 1;
      if (li >= 2) mbar_wait(&B.acc_empty[buf], ((li >> 1) - 1) & 1);
      tc_fence_after();
      mbar_wait(&B.full[s], (li / NST) & 1);
      tc_fence_after();
      const uint32_t st = su + O_ST + s * STB;
      if (elect_one()) {
#pragma unroll
        for (int c = 0; c < NS; ++c) {
          const uint64_t da = desc_k128(st + c * TS), db = desc_k128(su + O_B + c * BS);
#pragma unroll
          for (int ks = 0; ks < 4; ++ks) umma_ss(tmem + buf * NCOL + c * NB, da + ks * 2, db + ks * 2, IDESC, ks > 0 ? 1u : 0u);
        }
        tc_commit(&B.acc[buf]);
      }
      __syncwarp();
    }
  } else if (warp >= 8) {
    const uint32_t r = (uint32_t)((warp & 3) * 32 + lane);
    for (int li = 0; li < my; ++li) {
      const int s = li % NST, buf = li & 1;
      const uint32_t sx = su + O_ST + s * STB;
      if (li >= 2) mbar_wait(&B.sempty[buf], ((li >> 1) - 1) & 1);
      mbar_wait(&B.full[s], (li / NST) & 1);
      float s1a = 0.f, s1b = 0.f;
      SUNR_PRAGMA
      for (int c = 0; c < NS; ++c)
#pragma unroll
        for (int q = 0; q < 8; q += 2) {
          s1a += sum8(lds128(sx + c * TS + sw128(r, q)));
          s1b += sum8(lds128(sx + c * TS + sw128(r, q + 1)));
        }
      const float mean = (s1a + s1b) * (1.f / CP);
      float s2a = 0.f, s2b = 0.f;
      SUNR_PRAGMA
      for (int c = 0; c < NS; ++c)
#pragma unroll
        for (int q = 0; q < 8; q += 2) {
          s2a += sq8(lds128(sx + c * TS + sw128(r, q)), mean);
          s2b += sq8(lds128(sx + c * TS + sw128(r, q + 1)), mean);
        }
      const float rs = rsqrtf((s2a + s2b) * (1.f / CP) + EPSV);
      B.st[buf][r] = make_float2(mean, rs);
      mbar_arrive(&B.sfull[buf]);
    }
  } else if (warp >= 4) {
    const uint32_t lb = (uint32_t)(warp & 3) * 32, r = lb + lane, trow = tmem + (lb << 16);
    float wsum[16];
#pragma unroll
    for (int h = 0; h < 16; ++h) wsum[h] = B.wsum[h];
    for (int li = 0; li < my; ++li) {
      const int s = li % NST, buf = li & 1;
      const long r0 = (long)(vb + li * vg) * QM;
      const uint32_t sx = su + O_ST + s * STB;
      mbar_wait(&B.sfull[buf], (li >> 1) & 1);
      const float2 ms = B.st[buf][r];
      mbar_arrive(&B.sempty[buf]);
      const float mean = ms.x, rs = ms.y;
      const float mr = mean * rs;
      const bool fast = mr * mr <= 0.25f;
      float out[16];
      mbar_wait(&B.acc[buf], (li >> 1) & 1);
      tc_fence_after();
#pragma unroll
      for (int h = 0; h < 16; ++h) out[h] = 0.f;
#pragma unroll
      for (int c = 0; c < NS; ++c) {
        uint32_t lo[16], mid[16], hi[16];
        tmem_ld16(trow + buf * NCOL + c * NB, lo);
        tmem_ld16(trow + buf * NCOL + c * NB + 16, mid);
        tmem_ld16(trow + buf * NCOL + c * NB + 32, hi);
        tmem_wait_ld();
#pragma unroll
        for (int h = 0; h < 16; ++h) out[h] += (__uint_as_float(lo[h]) + __uint_as_float(mid[h])) + __uint_as_float(hi[h]);
      }
#pragma unroll
      for (int h = 0; h < 16; ++h) out[h] = rs * (out[h] - mean * wsum[h]);
      if (!fast) {   // the LN0-first path of this row (the TMEM reads above were warp-wide)
#pragma unroll
        for (int h = 0; h < 16; ++h) out[h] = 0.f;
#pragma unroll 1
        for (int c = 0; c < NS; ++c)
#pragma unroll 1
          for (int q = 0; q < 8; ++q) {
            const uint4 v = lds128(sx + c * TS + sw128(r, q));
            const float xv[8] = {bf16lo(v.x), bf16hi(v.x), bf16lo(v.y), bf16hi(v.y), bf16lo(v.z), bf16hi(v.z), bf16lo(v.w), bf16hi(v.w)};
#pragma unroll
            for (int e = 0; e < 8; ++e) {
              const float ln = (xv[e] - mean) * rs;
              const int col = c * 64 + q * 8 + e;
#pragma unroll
              for (int h = 0; h < 16; ++h) out[h] = fmaf(ln, __ldg(WF + h * CP + col), out[h]);
            }
          }
      }
      tc_fence_before();
      __syncwarp();
      if (lane == 0) { mbar_arrive(&B.acc_empty[buf]); mbar_arrive(&B.empty[s]); }
#pragma unroll
      for (int h = 0; h < 16; ++h) LG[(long)h * R + r0 + r] = out[h];
      if (use_pst) PST[r0 + r] = make_float2(mean, rs);
    }
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 2) { tc_fence_after(); tmem_dealloc(tmem, TCOLS); }
}
