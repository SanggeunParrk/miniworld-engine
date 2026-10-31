// tg_fused_f32.cu -- the whole fp32 (TF32) Transition forward at d = 64 / 128 in one tcgen05 kernel (inference):
//
//   y = x + Ws (silu(Wa xn) * (Wb xn)),   xn = tf32(LN(x) gamma + beta)
//
// One CTA per SM, persistent over 128-row tiles. Roles (448 threads):
//   warp 0       TMA producer: per 32-unit hidden chunk c the expand pair [Wa_c ; Wb_c] (one 64-row B operand, 3-stage ring) and the
//                squeeze chunk Ws[:, c] (2-stage ring). Every tile re-streams the weights from L2.
//   warp 1       MMA issuer (and TMEM owner): [a_c | b_c] = xn [Wa_c ; Wb_c]^T (kind::tf32, A = xn from TENSOR memory, N = 64,
//                double-buffered accumulators), then y += h_{c-1} Ws_{c-1}^T (A = h from shared memory) one chunk behind.
//   warps 2-9    epilogue: h_c = tf32(silu(a_c) b_c) into a K-major, 128-B swizzled shared-memory buffer (double-buffered); at the end
//                of a tile y = acc + x, staged in the idle h buffers and written by TMA.
//   warps 10-13  LayerNorm: one row per thread (its TMEM lane), statistics in fp32 over three passes of the L1-resident row, xn written
//                to tensor memory (tcgen05.st) as the next tile's A operand -- double-buffered, so the LayerNorm of tile i + 1 overlaps the
//                products of tile i and neither xn nor h reaches HBM (the generic chain wrote xn and h in fp32 and read them back).
// TMEM columns: [a | b] x 2 at 0 / 64, y at 128 (D columns), xn x 2 at 256 / 384.
// SPDX-License-Identifier: Apache-2.0
#include "sm100.cuh"
using namespace s100;

#ifndef D
#define D 128
#endif
#ifndef BN
#define BN 32                                    // hidden units per chunk (one 128-byte K-block of h)
#endif
constexpr int NSW = 3, NSS = 2, NH = 2;          // stages: expand weight chunks, squeeze weight chunks, h buffers
constexpr int KBD = D / 32;                      // 128-byte K-blocks of a D-wide fp32 row
constexpr int WKB = 2 * BN * 128;                // one K-block of a ring stage: Wa_c rows then Wb_c rows
constexpr int WST = KBD * WKB;                   // a ring stage: [Wa_c ; Wb_c] [2 BN][D]
constexpr int WS_BYTES = D * 128;                // the squeeze chunk Ws[:, c] as [D][BN = 32]: one K-block
constexpr int H_BYTES = 128 * 128;               // h_c [128][32] fp32: one K-block
constexpr int O_W = 0, O_WS = O_W + NSW * WST, O_H = O_WS + NSS * WS_BYTES;
constexpr int O_BAR = O_H + NH * H_BYTES;
constexpr int SMEM_BYTES = O_BAR + 1024;
static_assert(SMEM_BYTES <= 232448, "shared memory");
static_assert(D == 64 || D == 128, "width");
static_assert(BN == 32, "chunk = one K-block of h");
constexpr int NEPI = 8, NLN = 4;
constexpr int HALF = BN / 2;                     // columns of a chunk per epilogue warp
constexpr int T_AB = 0, T_Y = 128, T_X = 256;

// kind::tf32 instruction descriptor: TF32 A / B, fp32 D, K-major both
__host__ __device__ constexpr uint32_t idesc_tf32(int M, int N) {
  return (1u << 4) | (2u << 7) | (2u << 10) | ((uint32_t)(N >> 3) << 17) | ((uint32_t)(M >> 4) << 24);
}
DEVI void umma_t(uint32_t d, uint64_t a, uint64_t b, uint32_t id, uint32_t acc) {
  asm volatile("{ .reg .pred p; setp.ne.b32 p, %4, 0; tcgen05.mma.cta_group::1.kind::tf32 [%0], %1, %2, %3, p; }"
               :: "r"(d), "l"(a), "l"(b), "r"(id), "r"(acc) : "memory");
}
DEVI void umma_tt(uint32_t d, uint32_t a_tmem, uint64_t b, uint32_t id, uint32_t acc) {
  asm volatile("{ .reg .pred p; setp.ne.b32 p, %4, 0; tcgen05.mma.cta_group::1.kind::tf32 [%0], [%1], %2, %3, p; }"
               :: "r"(d), "r"(a_tmem), "l"(b), "r"(id), "r"(acc) : "memory");
}
DEVI float tf32r(float x) { uint32_t r; asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(x)); return __uint_as_float(r); }

struct Bars {
  uint64_t yfull, yempty;
  uint64_t xfull[2], xempty[2];
  uint64_t wfull[NSW], wempty[NSW], sfull[NSS], sempty[NSS];
  uint64_t abfull[2], abempty[2], hfull[NH], hempty[NH];
  uint32_t tmem;
};

extern "C" __global__ void __launch_bounds__(64 + 32 * (NEPI + NLN), 1)
tg_fused_f32_sm100(const __grid_constant__ CUtensorMap mWa, const __grid_constant__ CUtensorMap mWb, const __grid_constant__ CUtensorMap mWs,
                   const __grid_constant__ CUtensorMap mY, const float* __restrict__ X, const float* __restrict__ G, const float* __restrict__ Bt,
                   float eps, int M, int HID) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  Bars& B = *reinterpret_cast<Bars*>(sm + O_BAR);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int tiles = (M + 127) / 128, NC = HID / BN;

  if (threadIdx.x == 0) {
    mbar_init(&B.yfull, 1); mbar_init(&B.yempty, NEPI);
    for (int s = 0; s < 2; ++s) { mbar_init(&B.xfull[s], NLN); mbar_init(&B.xempty[s], 1); }
    for (int s = 0; s < NSW; ++s) { mbar_init(&B.wfull[s], 1); mbar_init(&B.wempty[s], 1); }
    for (int s = 0; s < NSS; ++s) { mbar_init(&B.sfull[s], 1); mbar_init(&B.sempty[s], 1); }
    for (int s = 0; s < 2; ++s) { mbar_init(&B.abfull[s], 1); mbar_init(&B.abempty[s], NEPI); }
    for (int s = 0; s < NH; ++s) { mbar_init(&B.hfull[s], NEPI); mbar_init(&B.hempty[s], 1); }
    fence_barrier_init();
    prefetch_map(&mWa); prefetch_map(&mWb); prefetch_map(&mWs); prefetch_map(&mY);
  }
  if (warp == 1) { tmem_alloc(smem_u32(&B.tmem), 512); tmem_relinquish(); }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = B.tmem;

  if (warp == 0) {
    // ------------------------------------------------------------------------------------------------ producer
    if (lane == 0) {
      int g = 0;
      for (int t = blockIdx.x; t < tiles; t += gridDim.x) {
        for (int c = 0; c < NC; ++c, ++g) {
          const int sw = g % NSW, ss = g % NSS;
          mbar_wait(&B.wempty[sw], ((g / NSW) & 1) ^ 1);
          mbar_expect_tx(&B.wfull[sw], WST);
          const uint32_t wb = su + O_W + sw * WST;
          for (int k = 0; k < KBD; ++k) {
            tma_load_2d(wb + k * WKB, &mWa, &B.wfull[sw], k * 32, c * BN);
            tma_load_2d(wb + k * WKB + BN * 128, &mWb, &B.wfull[sw], k * 32, c * BN);
          }
          mbar_wait(&B.sempty[ss], ((g / NSS) & 1) ^ 1);
          mbar_expect_tx(&B.sfull[ss], WS_BYTES);
          tma_load_2d(su + O_WS + ss * WS_BYTES, &mWs, &B.sfull[ss], c * BN, 0);
        }
      }
    }
  } else if (warp == 1) {
    // ------------------------------------------------------------------------------------------------ MMA issuer
    if (lane == 0) {
      constexpr uint32_t IDE = idesc_tf32(128, 2 * BN), IDS = idesc_tf32(128, D);
      int tc = 0, g = 0;
      for (int t = blockIdx.x; t < tiles; t += gridDim.x, ++tc) {
        const int xb = tc & 1;
        mbar_wait(&B.xfull[xb], (tc >> 1) & 1);
        tc_fence_after();
        const uint32_t tx = tmem + T_X + xb * 128;
        for (int c = 0; c <= NC; ++c) {
          if (c < NC) {                                         // expand of chunk c
            const int gc = g + c, buf = gc & 1, sw = gc % NSW;
            mbar_wait(&B.abempty[buf], ((gc >> 1) & 1) ^ 1);
            mbar_wait(&B.wfull[sw], (gc / NSW) & 1);
            tc_fence_after();
            const uint32_t ta = tmem + T_AB + buf * 2 * BN, wb = su + O_W + sw * WST;
            for (int k = 0; k < KBD; ++k) {
              const uint64_t dw = desc_k128(wb + k * WKB);
#pragma unroll
              for (int kk = 0; kk < 4; ++kk) umma_tt(ta, tx + k * 32 + kk * 8, dw + (uint64_t)(kk * 2), IDE, (k > 0 || kk > 0) ? 1u : 0u);
            }
            tc_commit(&B.wempty[sw]);
            tc_commit(&B.abfull[buf]);
            if (c == NC - 1) tc_commit(&B.xempty[xb]);
          }
          if (c >= 1) {                                         // squeeze of chunk c - 1
            const int gc = g + c - 1, hb = gc % NH, ss = gc % NSS;
            if (c == 1) mbar_wait(&B.yempty, (tc & 1) ^ 1);
            mbar_wait(&B.hfull[hb], (gc / NH) & 1);
            mbar_wait(&B.sfull[ss], (gc / NSS) & 1);
            tc_fence_after();
            const uint64_t dh = desc_k128(su + O_H + hb * H_BYTES), dw = desc_k128(su + O_WS + ss * WS_BYTES);
#pragma unroll
            for (int kk = 0; kk < 4; ++kk)
              umma_t(tmem + T_Y, dh + (uint64_t)(kk * 2), dw + (uint64_t)(kk * 2), IDS, (c > 1 || kk > 0) ? 1u : 0u);
            tc_commit(&B.sempty[ss]);
            tc_commit(&B.hempty[hb]);
            if (c == NC) tc_commit(&B.yfull);
          }
        }
        g += NC;
      }
    }
  } else if (warp >= 2 + NEPI) {
    // ------------------------------------------------------------------------------------------------ LayerNorm (warps 10-13)
    const uint32_t lrow = (uint32_t)((warp & 3) * 32) << 16;
    const int rr = (warp & 3) * 32 + lane;
    int tc = 0;
    for (int t = blockIdx.x; t < tiles; t += gridDim.x, ++tc) {
      const int xb = tc & 1;
      mbar_wait(&B.xempty[xb], ((tc >> 1) & 1) ^ 1);
      tc_fence_after();
      const long row = (long)t * 128 + rr;
      const bool ok = row < M;
      const float4* xr = reinterpret_cast<const float4*>(X + (ok ? row : 0) * D);
      float s = 0.f;
#pragma unroll 8
      for (int i = 0; i < D / 4; ++i) { const float4 v = __ldg(xr + i); s += (v.x + v.y) + (v.z + v.w); }
      const float mean = s / (float)D;
      float q = 0.f;
#pragma unroll 8
      for (int i = 0; i < D / 4; ++i) {
        const float4 v = __ldg(xr + i);
        const float a = v.x - mean, b = v.y - mean, c = v.z - mean, e = v.w - mean;
        q += (a * a + b * b) + (c * c + e * e);
      }
      const float rstd = rsqrtf(q / (float)D + eps);
#pragma unroll
      for (int u = 0; u < D / 16; ++u) {
        uint32_t o[16];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const float4 v = __ldg(xr + 4 * u + j);
          const float4 gg = __ldg(reinterpret_cast<const float4*>(G) + 4 * u + j), bb = __ldg(reinterpret_cast<const float4*>(Bt) + 4 * u + j);
          o[4 * j] = __float_as_uint(tf32r((v.x - mean) * rstd * gg.x + bb.x));
          o[4 * j + 1] = __float_as_uint(tf32r((v.y - mean) * rstd * gg.y + bb.y));
          o[4 * j + 2] = __float_as_uint(tf32r((v.z - mean) * rstd * gg.z + bb.z));
          o[4 * j + 3] = __float_as_uint(tf32r((v.w - mean) * rstd * gg.w + bb.w));
        }
        tmem_st16(tmem + lrow + T_X + xb * 128 + 16 * u, o);
      }
      tmem_wait_st();
      tc_fence_before();
      __syncwarp();
      if (lane == 0) mbar_arrive(&B.xfull[xb]);
    }
  } else {
    // ------------------------------------------------------------------------------------------------ epilogue (warps 2-9)
    const int ew = warp - 2, q4 = warp & 3, half = ew >> 2;
    const uint32_t r = (uint32_t)(q4 * 32 + lane);           // row within the tile
    const uint32_t lrow = (uint32_t)(q4 * 32) << 16;
    int tc = 0, g = 0;
    for (int t = blockIdx.x; t < tiles; t += gridDim.x, ++tc) {
      for (int c = 0; c < NC; ++c, ++g) {
        const int buf = g & 1, hb = g % NH;
        mbar_wait(&B.abfull[buf], (g >> 1) & 1);
        tc_fence_after();
        uint32_t a[HALF], b[HALF];
        tmem_ld16(tmem + lrow + T_AB + buf * 2 * BN + half * HALF, a);
        tmem_ld16(tmem + lrow + T_AB + buf * 2 * BN + BN + half * HALF, b);
        tmem_wait_ld();
        tc_fence_before();
        __syncwarp();
        if (lane == 0) mbar_arrive(&B.abempty[buf]);
        float h[HALF];
#pragma unroll
        for (int e = 0; e < HALF; ++e) {
          const float av = __uint_as_float(a[e]);
          h[e] = tf32r(av * sigmoid_kit(av) * __uint_as_float(b[e]));
        }
        mbar_wait(&B.hempty[hb], ((g / NH) & 1) ^ 1);
        const uint32_t hbase = su + O_H + hb * H_BYTES;
#pragma unroll
        for (int q = 0; q < HALF / 4; ++q)
          sts128(hbase + sw128(r, half * (HALF / 4) + q), make_uint4(__float_as_uint(h[4 * q]), __float_as_uint(h[4 * q + 1]),
                                                                     __float_as_uint(h[4 * q + 2]), __float_as_uint(h[4 * q + 3])));
        fence_proxy_async();
        __syncwarp();
        if (lane == 0) mbar_arrive(&B.hfull[hb]);
      }
      // y = acc + x for this warp's 32 rows x D / 2 columns, staged 32 columns at a time in the idle h buffers (4 KB per warp, 128-B
      // swizzled) and written by TMA
      mbar_wait(&B.yfull, tc & 1);
      tc_fence_after();
      const long row = (long)t * 128 + r;
      const bool ok = row < M;
      const uint32_t slab = su + O_H + (uint32_t)ew * 4096u;
#pragma unroll
      for (int u = 0; u < D / 64; ++u) {
        const int col = half * (D / 2) + u * 32;
        uint32_t acc[32];
        tmem_ld32(tmem + lrow + T_Y + col, acc);
        float xv[32];
        const float* xp = X + row * D + col;
#pragma unroll
        for (int q = 0; q < 8; ++q) {
          const uint4 w = ok ? ldg128(xp + 4 * q) : make_uint4(0, 0, 0, 0);
          xv[4 * q] = __uint_as_float(w.x); xv[4 * q + 1] = __uint_as_float(w.y); xv[4 * q + 2] = __uint_as_float(w.z); xv[4 * q + 3] = __uint_as_float(w.w);
        }
        tmem_wait_ld();
        if (u > 0) { if (lane == 0) tma_store_wait_read0(); __syncwarp(); }
#pragma unroll
        for (int q = 0; q < 8; ++q)
          sts128(slab + sw128((uint32_t)lane, q), make_uint4(__float_as_uint(__uint_as_float(acc[4 * q]) + xv[4 * q]),
                                                             __float_as_uint(__uint_as_float(acc[4 * q + 1]) + xv[4 * q + 1]),
                                                             __float_as_uint(__uint_as_float(acc[4 * q + 2]) + xv[4 * q + 2]),
                                                             __float_as_uint(__uint_as_float(acc[4 * q + 3]) + xv[4 * q + 3])));
        fence_proxy_async();
        __syncwarp();
        if (lane == 0) { tma_store_2d(&mY, slab, col, t * 128 + q4 * 32); tma_store_commit(); }
      }
      tc_fence_before();
      __syncwarp();
      if (lane == 0) mbar_arrive(&B.yempty);
      // the h buffers are written again by the next tile's chunks: every warp's staged stores must have been read out
      if (lane == 0) tma_store_wait_read0();
      named_bar_sync(1, 32 * NEPI);
    }
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 1) { tc_fence_after(); tmem_dealloc(tmem, 512); }
}

extern "C" __device__ int transition_smem_bytes = SMEM_BYTES;
