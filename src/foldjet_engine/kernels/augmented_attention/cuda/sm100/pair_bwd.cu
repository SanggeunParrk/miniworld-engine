// pair_bwd.cu -- AttentionPairBias pair backward on sm_100a (tcgen05), d_pair 128, 2026-11-17 (``sm100.apb_pair_bwd``; -DPB_NH = 8 |
// 12 | 16 | 24). What apb_rows.cu pair_bias_bwd_k computes (the mma.sync pass the B200 ran), with every product on the tensor core:
//
//   x^ = (x - mean) rstd (LN of a pair row, its weight folded into Wf), a = rstd dbias (bf16, the operand of both products below)
//   dpair[c]  = sum_h a_h Wf[h, c] + k x[c] + c0      k = -rstd^2 m2, c0 = rstd (rstd m2 mean - m1),
//               m1 = sum_h dbias_h s_h / 128, m2 = rstd (S2 - mean S1) / 128, S1 = sum_h dbias_h s_h, S2 = sum_h dbias_h (Wf . x)_h
//   dWf[h, c] += sum_r a[r, h] x[r, c] - sum_r a[r, h] mean_r,  hsum[h] += sum_r dbias[r, h]       (s_h = sum_c Wf[h, c])
//
// Per tile (128 consecutive keys of one query row; L % 128 == 0), a ring of NS stages holds the pair rows (two TMA boxes of 64 columns,
// 128-B swizzle) and dbias [NH][128] fp32. Roles (352 threads, one block per SM):
//   warp 9   loader: a tile into a stage once the store warp has handed the stage back;
//   warp 8   MMA issuer, whichever is ready first:
//            MMA1(n) = X [Wf ; 1]^T (Wf . x per head and the row sum; N = 32);
//            MMA2(n) = D[n & 1] = a Wf (A = a[n & 1] [128 keys][64 heads] K-major, B = Wf^T [128 columns][64 heads] K-major, K = 16 / 32)
//                      and dWf^T += X^T a (A = the pair tile read MN-major, B = a read MN-major, N = 64; in TMEM over all the block's tiles);
//   warps 0-7 epilogue, thread = key, warps w and w + 4 share TMEM lanes 32 (w % 4) and take one 64-column box each. Phase A(n): the
//            row's sum of squares from shared memory (the halves' sums exchanged; no Gram MMA: it cost 0.5 us of tensor time per
//            tile on the critical path), mean from MMA1, S1, S2, the halves' 8-head chunks of a into a[n & 1]. Phase B(n): dpair = D + k x
//            + c0 written over the pair tile in place. A(n + 1) runs before B(n), so MMA2(n + 1) overlaps B(n);
//   warp 10  store warp: TMA-stores both dpair boxes of tile n, waits for their read, hands the stage back (the epilogue never waits).
// dWf, the mean term and hsum leave as the block's partial row (the caller's accumulator row blockIdx.x, reduced in a fixed order by
// finalize). Wf's global reads go out before the first tiles' loads (behind 148 SMs' tiles they took ~5 us of start-up).
// Measured (2026-11-17, alone, graph replay, L2 flushed): 16 heads L 512 43.0 -> 38.9 us, 24 heads 51.2 -> 49.2, 12 heads L 128 12.3 ->
// 12.3, 8 heads L 768 63.5 -> 65.5 (the shared pass runs two blocks per SM there); a block moves ~5.3 TB/s of the card's rate in steady state.
// SPDX-License-Identifier: Apache-2.0
#include "sm100.cuh"
using namespace s100;

#ifndef PB_NH
#define PB_NH 16
#endif
constexpr int NH = PB_NH, DP = 128, TR = 128;
constexpr int NHC = (NH + 7) / 8, KH = (NH + 15) / 16;                  // 8-head chunks of a row of a; K = 16 steps of D = a Wf
#ifndef PB_SPLIT
#define PB_SPLIT 1                       // a = a_hi + a_lo (two bf16 parts, the low one at head slots LO..): D and dWf see a at ~fp32 precision
#endif
constexpr int LO = NH <= 16 ? 16 : 32, KD = PB_SPLIT ? (LO + NH + 15) / 16 : KH;   // K steps of D = a Wf (with the low part)
constexpr int ONES = 24, NB32 = 32;                                     // B rows of MMA1: heads, then ones at row 24
constexpr int XB = TR * DP * 2, BOX = TR * 128, DBB = NH * TR * 4;
constexpr int STG = XB + ((DBB + 1023) / 1024) * 1024, NS = NH > 16 ? 3 : 4;
constexpr int O_A = NS * STG, O_WT = O_A + 2 * TR * 128, O_WF = O_WT + DP * 128, O_MISC = O_WF + 2 * NB32 * 128;
constexpr int SMEM_BYTES = O_MISC + 4096;
static_assert(SMEM_BYTES <= 232448 && NH <= ONES, "shared memory / heads");
constexpr uint32_t T_Y = 0, T_D = 64, T_W = 320;                       // TMEM columns: [Wf ; 1] x, D (two: 64, 192), dWf^T
struct Misc {
  uint64_t full[NS], empty[NS], edone[NS], gdone, gfree, afull[2], ddone[2];
  float sw[NB32];
  float tot[2 * NH];
  float ssq[2][2][TR];                                                    // [tile parity][half][key]: the halves' sums of squares
  uint32_t tmem;
};

extern "C" __global__ void __launch_bounds__(352, 1) apb_pair_bwd_sm100(const __grid_constant__ CUtensorMap mz,
    const __grid_constant__ CUtensorMap mdz, const __grid_constant__ CUtensorMap md, const __nv_bfloat16* __restrict__ W,
    float* __restrict__ AWF, int acc_stride, int L, float eps, unsigned long long* __restrict__ TRC) {
#define TRK(ev, n) do { if (TRC != nullptr && blockIdx.x == 0 && (n) < 64) TRC[(ev) * 64 + (n)] = clock64(); } while (0)
  extern __shared__ __align__(1024) uint8_t smem[];
  Misc& B = *reinterpret_cast<Misc*>(smem + O_MISC);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const uint32_t su = smem_u32(smem);
  auto gtime = []() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; };
  if (TRC != nullptr && threadIdx.x == 0) TRC[384 + 4 * blockIdx.x] = gtime();
  const int ntj = L / TR;
  const long ntiles = (long)L * ntj;
  const long mine = blockIdx.x < ntiles ? (ntiles - blockIdx.x + gridDim.x - 1) / gridDim.x : 0;
  auto issue = [&](long n) {
    const long t = blockIdx.x + n * gridDim.x;
    const int i = (int)(t / ntj), j0 = (int)(t % ntj) * TR, s = (int)(n % NS);
    const uint32_t st = su + s * STG;
    mbar_expect_tx(&B.full[s], (uint32_t)(XB + DBB));
    tma_load_2d(st, &mz, &B.full[s], 0, i * L + j0);
    tma_load_2d(st + BOX, &mz, &B.full[s], 64, i * L + j0);
    tma_load_3d(st + XB, &md, &B.full[s], j0, i, 0);
  };
  uint4 wv[2];                                                             // Wf's words, read before the tiles' loads queue at the memory
#pragma unroll
  for (int k = 0; k < 2; ++k) {
    const int e = tid + 352 * k, r = e >> 4, c0 = (e & 15) * 8;
    const uint32_t one = r == ONES ? 0x3f803f80u : 0u;
    wv[k] = e < NB32 * DP / 8 && r < NH ? ldg128(W + r * DP + c0) : make_uint4(one, one, one, one);
  }
  if (tid == 0) {
    for (int s = 0; s < NS; ++s) { mbar_init(&B.full[s], 1); mbar_init(&B.empty[s], 1); mbar_init(&B.edone[s], 8); }
    mbar_init(&B.gdone, 1); mbar_init(&B.gfree, 8);
    for (int k = 0; k < 2; ++k) { mbar_init(&B.afull[k], 8); mbar_init(&B.ddone[k], 1); }
    fence_barrier_init();
  }
  if (warp == 0) { tmem_alloc(smem_u32(&B.tmem), 512); tmem_relinquish(); }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  if (tid == 0)
    for (long n = 0; n < NS && n < mine; ++n) issue(n);
  // operands made once: the MMA1 B boxes ([Wf ; 1], rows 0..31 x 64 columns per box), Wf^T [128 columns][64 heads], a's zero heads,
  // s_h = sum_c Wf[h, c] (every global read issued before the first is used)
  {
    uint8_t* wf = smem + O_WF;
    uint8_t* wt = smem + O_WT;
    uint8_t* at = smem + O_A;
    for (int e = tid; e < 2 * TR * 8; e += 352) *reinterpret_cast<uint4*>(at + 16 * e) = make_uint4(0u, 0u, 0u, 0u);
    for (int e = tid; e < DP * 8; e += 352) *reinterpret_cast<uint4*>(wt + 16 * e) = make_uint4(0u, 0u, 0u, 0u);
    __syncthreads();
    // thread e (of NB32 DP / 8 = 512, two passes) takes 8 consecutive columns of row r = e / 16: both passes' reads go out first
#pragma unroll
    for (int k = 0; k < 2; ++k) {
      const int e = tid + 352 * k, r = e >> 4, c0 = (e & 15) * 8;
      float s = 0.f;
      if (e < NB32 * DP / 8) {
        *reinterpret_cast<uint4*>(wf + (c0 >> 6) * (NB32 * 128) + sw128((uint32_t)r, (uint32_t)((c0 & 63) >> 3))) = wv[k];
        const uint32_t w4[4] = {wv[k].x, wv[k].y, wv[k].z, wv[k].w};
        if (r < NH) {
#pragma unroll
          for (int q = 0; q < 8; ++q) {
            const uint32_t u = w4[q >> 1];
            *reinterpret_cast<uint16_t*>(wt + sw128((uint32_t)(c0 + q), (uint32_t)(r >> 3)) + (r & 7) * 2) = (uint16_t)((q & 1) ? (u >> 16) : (u & 0xffffu));
            if (PB_SPLIT) *reinterpret_cast<uint16_t*>(wt + sw128((uint32_t)(c0 + q), (uint32_t)((LO + r) >> 3)) + ((LO + r) & 7) * 2) = (uint16_t)((q & 1) ? (u >> 16) : (u & 0xffffu));
            s += (q & 1) ? bf16hi(u) : bf16lo(u);
          }
        }
      }
#pragma unroll
      for (int o = 8; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);   // the 16 threads of row r (16-aligned in the warp)
      if ((tid & 15) == 0 && e < NB32 * DP / 8) B.sw[r] = s;
    }
  }
  fence_proxy_async();
  __syncthreads();
  const uint32_t tmem = B.tmem;
  if (TRC != nullptr && threadIdx.x == 0) TRC[384 + 4 * blockIdx.x + 1] = gtime();
  if (warp == 8) {
    if (lane == 0) {                                                       // the MMA thread
      constexpr uint32_t I_Y = idesc_bf16(TR, NB32), I_D = idesc_bf16(TR, DP), I_W = idesc_bf16(DP, 64, 1, 1);
      const uint64_t dwt = desc_k128(su + O_WT);
      auto mma1 = [&](long n) {                                           // [Wf ; 1] x of tile n
        const int s = (int)(n % NS);
        const uint32_t st = su + s * STG;
        TRK(0, n);
        tc_fence_after();
#pragma unroll
        for (int b = 0; b < 2; ++b) {
          const uint64_t dx = desc_k128(st + b * BOX), dw = desc_k128(su + O_WF + b * NB32 * 128);
#pragma unroll
          for (int ks = 0; ks < 4; ++ks) {
            umma_ss(tmem + T_Y, dx + (uint64_t)(2 * ks), dw + (uint64_t)(2 * ks), I_Y, (b | ks) ? 1u : 0u);
          }
        }
        tc_commit(&B.gdone);
      };
      // MMA1(n1) once tile n1 is in and phase A(n1 - 1) has read the Wf . x columns; MMA2(n2) once a of tile n2 is in (and D[n2 & 1]
      // was read out): whichever is ready first, so neither waits behind the other
      long n1 = 0, n2 = 0;
      while (n2 < mine) {
        if (n1 < mine && mbar_test(&B.full[n1 % NS], (uint32_t)((n1 / NS) & 1)) && (n1 == 0 || mbar_test(&B.gfree, (uint32_t)((n1 - 1) & 1)))) {
          if (TRC != nullptr && n1 == 0) TRC[384 + 4 * blockIdx.x + 2] = gtime();
          mma1(n1++);
          continue;
        }
        if (n2 >= n1 || !mbar_test(&B.afull[n2 & 1], (uint32_t)((n2 >> 1) & 1))) continue;
        const long n = n2++;
        const int s = (int)(n % NS), ab = (int)(n & 1);
        const uint32_t st = su + s * STG;
        tc_fence_after();
        const uint64_t da = desc_k128(su + O_A + ab * TR * 128), dab = desc_mn128(su + O_A + ab * TR * 128, BOX);
#pragma unroll
        for (int ks = 0; ks < KD; ++ks) umma_ss(tmem + T_D + 128 * ab, da + (uint64_t)(2 * ks), dwt + (uint64_t)(2 * ks), I_D, ks ? 1u : 0u);
        const uint64_t dxm = desc_mn128(st, BOX);
#pragma unroll
        for (int ks = 0; ks < TR / 16; ++ks)                              // K = 16 keys: 2 KB of every box
          umma_ss(tmem + T_W, dxm + (uint64_t)((ks * 2048) >> 4), dab + (uint64_t)((ks * 2048) >> 4), I_W, (n | ks) ? 1u : 0u);
        tc_commit(&B.ddone[ab]);
      }
    }
  } else if (warp == 10) {
    if (lane == 0) {                                                       // the store warp: tile n's dpair boxes, then its stage back to the loader
      for (long n = 0; n < mine; ++n) {
        const long t = blockIdx.x + n * gridDim.x;
        const int s = (int)(n % NS), i = (int)(t / ntj), j0 = (int)(t % ntj) * TR;
        mbar_wait(&B.edone[s], (uint32_t)((n / NS) & 1));
        tma_store_2d(&mdz, su + s * STG, 0, i * L + j0);
        tma_store_2d(&mdz, su + s * STG + BOX, 64, i * L + j0);
        tma_store_commit();
        tma_store_wait_read0();
        mbar_arrive(&B.empty[s]);
      }
      tma_store_wait0();
    }
  } else if (warp == 9) {
    if (lane == 0)                                                         // the loader
      for (long n = NS; n < mine; ++n) {
        mbar_wait(&B.empty[n % NS], (uint32_t)(((n / NS) - 1) & 1));
        TRK(5, n);
        issue(n);
      }
  } else {                                                                 // the epilogue: thread = key r of the tile, half = column box
    const int q4 = warp & 3, half = warp >> 2, r = 32 * q4 + lane;
    const uint32_t trow = tmem + ((uint32_t)(32 * q4) << 16);
    float sw[NH];
#pragma unroll
    for (int h = 0; h < NH; ++h) sw[h] = B.sw[h];
    float corr[NH], hsum[NH];
#pragma unroll
    for (int h = 0; h < NH; ++h) corr[h] = hsum[h] = 0.f;
    // phase A of tile n: statistics, S1, S2, a into a[n & 1]; returns (k, c0) of dpair = D + k x + c0
    auto phase_a = [&](long n, float& kk, float& c0) {
      const int s = (int)(n % NS), ab = (int)(n & 1);
      const uint8_t* stg = smem + s * STG;
      mbar_wait(&B.gdone, (uint32_t)(n & 1));
      if (threadIdx.x == 0) TRK(1, n);
      tc_fence_after();
      uint32_t y[32];
      tmem_ld32(trow + T_Y, y);
      float sq = 0.f;                                                      // this half's sum of squares of the row (bf16 values, fp32)
      {
        const uint32_t box = su + s * STG + half * BOX;
#pragma unroll
        for (int q = 0; q < 8; ++q) {
          const uint4 xv = lds128(box + sw128((uint32_t)r, (uint32_t)q));
          const uint32_t xw[4] = {xv.x, xv.y, xv.z, xv.w};
#pragma unroll
          for (int e = 0; e < 4; ++e) { sq = fmaf(bf16lo(xw[e]), bf16lo(xw[e]), sq); sq = fmaf(bf16hi(xw[e]), bf16hi(xw[e]), sq); }
        }
      }
      B.ssq[ab][half][r] = sq;
      tmem_wait_ld();
      tc_fence_before();
      __syncwarp();
      if (lane == 0) mbar_arrive(&B.gfree);
      named_bar_sync(4, 256);
      const float ss = B.ssq[ab][0][r] + B.ssq[ab][1][r];                 // half 0 + half 1: the same sum in both halves
      const float mean = __uint_as_float(y[ONES]) * (1.f / DP);
      const float rstd = rsqrtf(fmaxf(ss * (1.f / DP) - mean * mean, 0.f) + eps);
      const float* dbs = reinterpret_cast<const float*>(stg + XB);         // [NH][128]
      float d[NH];
#pragma unroll
      for (int h = 0; h < NH; ++h) d[h] = dbs[h * TR + r];
      float S1 = 0.f, S2 = 0.f;
#pragma unroll
      for (int h = 0; h < NH; ++h) { S1 = fmaf(d[h], sw[h], S1); S2 = fmaf(d[h], __uint_as_float(y[h]), S2); }
      const float m1 = S1 * (1.f / DP), m2 = rstd * (S2 - mean * S1) * (1.f / DP);
      kk = -rstd * rstd * m2; c0 = rstd * (rstd * m2 * mean - m1);
      if (half == 0) {
#pragma unroll
        for (int h = 0; h < NH; ++h) {
          corr[h] = fmaf(rstd * d[h], mean, corr[h]);
          hsum[h] += d[h];
        }
      }
#pragma unroll
      for (int cg = half; cg < NHC; cg += 2) {                             // the halves write alternate 8-head chunks
        uint32_t p[4], pl[4];
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const int h0 = 8 * cg + 2 * e, h1 = h0 + 1;
          const float a0 = h0 < NH ? rstd * d[h0 < NH ? h0 : 0] : 0.f, a1 = h1 < NH ? rstd * d[h1 < NH ? h1 : 0] : 0.f;
          p[e] = pack_bf16(a0, a1);
          pl[e] = pack_bf16(a0 - bf16lo(p[e]), a1 - bf16hi(p[e]));
        }
        sts128(su + O_A + ab * TR * 128 + sw128((uint32_t)r, (uint32_t)cg), make_uint4(p[0], p[1], p[2], p[3]));
        if (PB_SPLIT) sts128(su + O_A + ab * TR * 128 + sw128((uint32_t)r, (uint32_t)(LO / 8 + cg)), make_uint4(pl[0], pl[1], pl[2], pl[3]));
      }
      fence_proxy_async();
      __syncwarp();
      if (lane == 0) mbar_arrive(&B.afull[ab]);
      if (threadIdx.x == 0) TRK(2, n);
    };
    // phase B of tile n: dpair = D + k x + c0 over this half's 64 columns, in place, then a TMA store of the box
    auto phase_b = [&](long n, float kk, float c0) {
      const long t = blockIdx.x + n * gridDim.x;
      const int s = (int)(n % NS), ab = (int)(n & 1), i = (int)(t / ntj), j0 = (int)(t % ntj) * TR;
      mbar_wait(&B.ddone[ab], (uint32_t)((n >> 1) & 1));
      if (threadIdx.x == 0) TRK(3, n);
      tc_fence_after();
      uint32_t v0[32], v1[32];
      tmem_ld32(trow + T_D + 128 * ab + 64 * half, v0);
      tmem_ld32(trow + T_D + 128 * ab + 64 * half + 32, v1);
      tmem_wait_ld();
      tc_fence_before();
      const uint32_t box = su + s * STG + half * BOX;
#pragma unroll
      for (int q = 0; q < 8; ++q) {
        const uint32_t addr = box + sw128((uint32_t)r, (uint32_t)q);
        const uint4 xv = lds128(addr);
        const uint32_t xw[4] = {xv.x, xv.y, xv.z, xv.w};
        uint32_t o[4];
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const int c = 8 * q + 2 * e;
          const uint32_t d0 = c < 32 ? v0[c] : v1[c - 32], d1 = c + 1 < 32 ? v0[c + 1] : v1[c + 1 - 32];
          o[e] = pack_bf16(fmaf(kk, bf16lo(xw[e]), __uint_as_float(d0) + c0), fmaf(kk, bf16hi(xw[e]), __uint_as_float(d1) + c0));
        }
        sts128(addr, make_uint4(o[0], o[1], o[2], o[3]));
      }
      fence_proxy_async();
      __syncwarp();
      if (lane == 0) mbar_arrive(&B.edone[s]);                           // the store warp takes it from here
      (void)i; (void)j0;
      if (threadIdx.x == 0) TRK(4, n);
    };
    // A(0), then A(n + 1) before B(n): tile n + 1's D = a Wf and dWf run on the tensor core while tile n's dpair is formed
    float kc = 0.f, cc = 0.f, kn = 0.f, cn = 0.f;
    if (mine > 0) phase_a(0, kc, cc);
    for (long n = 0; n < mine; ++n) {
      if (n + 1 < mine) phase_a(n + 1, kn, cn);
      phase_b(n, kc, cc);
      kc = kn; cc = cn;
    }
    // ---- the block's partial row: dWf[h, c] = (X^T a)[c, h] - sum_r a mean (half 0), hsum
    float* cr = reinterpret_cast<float*>(smem + O_A);                      // [2 NH][128]: both a tiles are idle now (every MMA has completed)
    if (half == 0) {
#pragma unroll
      for (int h = 0; h < NH; ++h) { cr[h * TR + r] = corr[h]; cr[(NH + h) * TR + r] = hsum[h]; }
    }
    named_bar_sync(3, 256);
    if (tid < 2 * NH) {
      float v = 0.f;
      for (int k = 0; k < TR; ++k) v += cr[tid * TR + k];
      B.tot[tid] = v;
    }
    named_bar_sync(3, 256);
    float* dst = AWF + (long)blockIdx.x * acc_stride;
    if (half == 0) {
      uint32_t wv[64];
      if (mine > 0) {
        tc_fence_after();
        tmem_ld32(trow + T_W, *reinterpret_cast<uint32_t(*)[32]>(wv));
        tmem_ld32(trow + T_W + 32, *reinterpret_cast<uint32_t(*)[32]>(wv + 32));
        tmem_wait_ld();
      }
#pragma unroll
      for (int h = 0; h < NH; ++h)
        dst[h * DP + r] = mine > 0 ? (PB_SPLIT ? __uint_as_float(wv[h]) + __uint_as_float(wv[LO + h]) : __uint_as_float(wv[h])) - B.tot[h] : 0.f;
      if (r < NH) dst[NH * DP + r] = B.tot[NH + r];
    }
  }
  if (TRC != nullptr && warp == 0 && lane == 0) TRC[384 + 4 * blockIdx.x + 3] = gtime();
  tc_fence_before();
  __syncthreads();
  if (warp == 0) { tc_fence_after(); tmem_dealloc(tmem, 512); }
}
