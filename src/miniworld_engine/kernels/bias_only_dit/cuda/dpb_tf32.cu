// dpb_tf32.cu -- the bias-only attention's bias gradient for the fp32 path, sm_100a, TF32 tensor cores (the fp32 twin of
// dpb_sm100.cu, training backward):
//
//   dP_h[i, j]  = sum_a sum_d do[a, i, h, d] v[a, j, h, d]          (the attention weights are shared by the A samples)
//   dbias_h     = P_h o (dP_h - D_h),   D_h[i] = sum_a dd[a, h, i]   (dd [A, H, L] fp32: per sample sum_d do o)
//
// do, v: [A L, *] fp32 rows (a, token), head h in columns DH h .. DH h + DH - 1 (v the v half of the v|g GEMM output, any row
// stride); P, dbias: [H L, L] fp32 (head-major, query rows). A work item is (head, 128-query tile, NJ-key tile); its K loop walks
// the samples: per sample the do tile [128 i][DH d] and the key tile [NJ j][DH d], both K-major, one M 128 x N NJ x K DH product
// (K = 8 per kind::tf32 MMA) into an NJ-column TMEM accumulator, double-buffered by item parity (the epilogue of item k runs
// under the K loop of item k + 1).
// A fp32 head row is DH x 4 bytes: columns 0-31 are one 128-B SW128 box (4 K steps), columns 32..DH-1 a second box -- SW64
// (16 channels, 2 K steps) for DH 48, SW128 (4 K steps) for DH 64, none for DH 32 -- as attn_inf_tf32.cu's q / k tiles.
// The epilogue streams P in [128 i][32 j] fp32 pieces (SW128, 16 KB) through a two-slot ring loaded by its own producer warp; each
// warp stages its 32 rows x 32 keys of dbias (fp32 SW128, 4 KB) and stores them by TMA, NSW staging tiles per warp.
//
// Shared memory (bytes; 232448 per CTA): P ring 2 x 16 KB, dbias staging 4 warps x NSW (2) x 4 KB = 32 KB, then NST stages of
// STB = (128 + NJ) x (128 + 4 WB) (the do tile and the key tile, both halves), NST = (232448 - 1024 - 64 KB) / STB >= 2
// (static_assert), 256 B of barriers. Every swizzled tile base is a multiple of 1 KB.
//   16 x 48: NJ 128 3 x 48 KB, NJ 192 2 x 60 KB, NJ 256 2 x 72 KB;   12 x 64 / 16 x 64: NJ 128 2 x 64 KB, NJ 192 2 x 80 KB;
//   24 x 32: NJ 128 5 x 32 KB, NJ 256 3 x 48 KB
// Tensor memory: 2 NJ columns (<= 512).
// Warps: 0 K-loop TMA producer, 1 MMA (whole warp waits, elect_one() issues), 2 TMEM allocator, 3 P producer, 4-7 epilogue (one
// query row per thread: 32 accumulator registers + 32 P values per piece; at most 128 registers).
// SPDX-License-Identifier: Apache-2.0
#include "sm100.cuh"
using namespace s100;

#ifndef NJ
#define NJ 128
#endif
#ifndef NHEAD
#define NHEAD 16
#endif
#ifndef DHEAD
#define DHEAD 48
#endif
constexpr int QM = 128, DH = DHEAD, NH = NHEAD, NQ = NJ / 32;
static_assert(DH == 32 || DH == 48 || DH == 64, "head width 32, 48 or 64");
static_assert(NJ % 32 == 0 && NJ <= 256 && 2 * NJ <= 512, "key tile: N <= 256, two accumulators in 512 columns");
constexpr int WB = DH - 32;                                   // channels of the second box: 0, 16, 32
constexpr int RA = 128, RB = WB * 4;                          // bytes per row of the first / second box
constexpr int TDA = QM * RA, TDB = QM * RB, TVA = NJ * RA, TVB = NJ * RB;
constexpr int S_DA = 0, S_DB = TDA, S_VA = TDA + TDB, S_VB = S_VA + TVA, STB = S_VB + TVB;
constexpr int PC = QM * 128;                                  // P piece [128 i][32 j] fp32 SW128: 16 KB
constexpr int SWB = 32 * 128, NSW = 2;                        // per-warp dbias staging [32 rows][32 cols] fp32 SW128
constexpr int O_P = 0, O_S = O_P + 2 * PC, O_ST = O_S + 4 * NSW * SWB;
constexpr int NST = (232448 - O_ST - 1024) / STB;
constexpr int O_BAR = O_ST + NST * STB, SMEM_BYTES = O_BAR + 256;
static_assert(NST >= 2 && SMEM_BYTES <= 232448, "shared memory");
static_assert(O_ST % 1024 == 0 && STB % 1024 == 0 && S_VA % 1024 == 0 && S_VB % 512 == 0 && S_DB % 1024 == 0, "alignment");
constexpr uint32_t IDESC = idesc_tf32(QM, NJ), TCOLS = 2 * NJ <= 256 ? 256 : 512;

struct Bars { uint64_t full[NST], empty[NST], pfull[2], pempty[2], acc_full[2], acc_empty[2]; uint32_t tmem; };

extern "C" __global__ void __launch_bounds__(256, 2)
bo_dpb_tf32_sm100(const __grid_constant__ CUtensorMap mdoa, const __grid_constant__ CUtensorMap mdob,
                  const __grid_constant__ CUtensorMap mva, const __grid_constant__ CUtensorMap mvb,
                  const __grid_constant__ CUtensorMap mp, const __grid_constant__ CUtensorMap mdb,
                  const float* __restrict__ dd, int L, int A) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  Bars& B = *reinterpret_cast<Bars*>(sm + O_BAR);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int mt = L / QM, nj = L / NJ, items = NH * mt * nj;
  const int my = (items > (int)blockIdx.x) ? (items - (int)blockIdx.x + (int)gridDim.x - 1) / (int)gridDim.x : 0;
  auto item_of = [&](int li, int& h, int& i0, int& j0) {
    const int wi = (int)blockIdx.x + li * (int)gridDim.x;
    j0 = (wi % nj) * NJ;
    const int r = wi / nj;
    i0 = (r % mt) * QM; h = r / mt;
  };

  if (tid == 0) {
    for (int s = 0; s < NST; ++s) { mbar_init(&B.full[s], 1); mbar_init(&B.empty[s], 1); }
    for (int e = 0; e < 2; ++e) {
      mbar_init(&B.pfull[e], 1); mbar_init(&B.pempty[e], 4);
      mbar_init(&B.acc_full[e], 1); mbar_init(&B.acc_empty[e], 4);
    }
    fence_barrier_init();
    prefetch_map(&mdoa); prefetch_map(&mva); prefetch_map(&mp); prefetch_map(&mdb);
    if (WB > 0) { prefetch_map(&mdob); prefetch_map(&mvb); }
  }
  if (warp == 2) { tmem_alloc(smem_u32(&B.tmem), TCOLS); tmem_relinquish(); }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = B.tmem;

  if (warp == 0) {
    // ------------------------------------------------------------------------------------------------ K-loop TMA producer
    if (lane == 0) {
      int g = 0;
      for (int li = 0; li < my; ++li) {
        int h, i0, j0; item_of(li, h, i0, j0);
        for (int a = 0; a < A; ++a, ++g) {
          const int s = g % NST;
          if (g >= NST) mbar_wait(&B.empty[s], ((g / NST) - 1) & 1);
          const uint32_t st = su + O_ST + s * STB;
          mbar_expect_tx(&B.full[s], STB);
          tma_load_2d(st + S_DA, &mdoa, &B.full[s], h * DH, a * L + i0);
          tma_load_2d(st + S_VA, &mva, &B.full[s], h * DH, a * L + j0);
          if (WB > 0) {
            tma_load_2d(st + S_DB, &mdob, &B.full[s], h * DH + 32, a * L + i0);
            tma_load_2d(st + S_VB, &mvb, &B.full[s], h * DH + 32, a * L + j0);
          }
        }
      }
    }
  } else if (warp == 1) {
    // ------------------------------------------------------------------------------------------------ MMA issuer
    int g = 0;
    for (int li = 0; li < my; ++li) {
      const int b = li & 1;
      if (li >= 2) mbar_wait(&B.acc_empty[b], ((li >> 1) - 1) & 1);
      tc_fence_after();
      const uint32_t d = tmem + (uint32_t)(b * NJ);
      for (int a = 0; a < A; ++a, ++g) {
        const int s = g % NST;
        mbar_wait(&B.full[s], (g / NST) & 1);
        tc_fence_after();
        const uint32_t st = su + O_ST + s * STB;
        const uint64_t da = desc_k128(st + S_DA), dv = desc_k128(st + S_VA);
        if (elect_one()) {
#pragma unroll
          for (int ks = 0; ks < 4; ++ks)
            umma_ss_tf32(d, da + (uint64_t)(ks * 2), dv + (uint64_t)(ks * 2), IDESC, (a > 0 || ks > 0) ? 1u : 0u);
          if (WB == 16) {
            const uint64_t db = desc_sw64(st + S_DB), vb = desc_sw64(st + S_VB);
#pragma unroll
            for (int ks = 0; ks < 2; ++ks) umma_ss_tf32(d, db + (uint64_t)(ks * 2), vb + (uint64_t)(ks * 2), IDESC, 1u);
          } else if (WB == 32) {
            const uint64_t db = desc_k128(st + S_DB), vb = desc_k128(st + S_VB);
#pragma unroll
            for (int ks = 0; ks < 4; ++ks) umma_ss_tf32(d, db + (uint64_t)(ks * 2), vb + (uint64_t)(ks * 2), IDESC, 1u);
          }
          tc_commit(&B.empty[s]);
          if (a == A - 1) tc_commit(&B.acc_full[b]);
        }
        __syncwarp();
      }
    }
  } else if (warp == 3) {
    // ------------------------------------------------------------------------------------------------ P producer
    if (lane == 0) {
      int e = 0;
      for (int li = 0; li < my; ++li) {
        int h, i0, j0; item_of(li, h, i0, j0);
        for (int q = 0; q < NQ; ++q, ++e) {
          const int slot = e & 1;
          if (e >= 2) mbar_wait(&B.pempty[slot], ((e >> 1) - 1) & 1);
          mbar_expect_tx(&B.pfull[slot], PC);
          tma_load_2d(su + O_P + slot * PC, &mp, &B.pfull[slot], j0 + q * 32, h * L + i0);
        }
      }
    }
  } else if (warp >= 4) {
    // ------------------------------------------------------------------------------------------------ epilogue
    const uint32_t lb = (uint32_t)(warp & 3) * 32, r = lb + lane, trow = tmem + (lb << 16);
    const uint32_t stg = su + O_S + (uint32_t)(warp & 3) * NSW * SWB;
    int e = 0, t = 0;
    for (int li = 0; li < my; ++li) {
      int h, i0, j0; item_of(li, h, i0, j0);
      const int b = li & 1;
      float Dr = 0.f;                                             // D[i]: the per-sample row sums, read while the products run
      for (int a = 0; a < A; ++a) Dr += __ldg(dd + ((long)a * NH + h) * L + i0 + r);
      mbar_wait(&B.acc_full[b], (li >> 1) & 1);
      tc_fence_after();
#pragma unroll 1
      for (int q = 0; q < NQ; ++q, ++e, ++t) {
        uint32_t v[32];
        tmem_ld32(trow + (uint32_t)(b * NJ + q * 32), v);
        const int slot = e & 1;
        mbar_wait(&B.pfull[slot], (e >> 1) & 1);
        uint4 p4[8];
#pragma unroll
        for (int c = 0; c < 8; ++c) p4[c] = lds128(su + O_P + slot * PC + sw128(r, c));
        __syncwarp();
        if (lane == 0) mbar_arrive(&B.pempty[slot]);
        tmem_wait_ld();
        if (q == NQ - 1) {                                        // the accumulator is read: the next-but-one item may use it
          tc_fence_before();
          __syncwarp();
          if (lane == 0) mbar_arrive(&B.acc_empty[b]);
        }
        const uint32_t sb = stg + (uint32_t)(t % NSW) * SWB;
#pragma unroll
        for (int c = 0; c < 8; ++c) {
          const uint4 o = make_uint4(__float_as_uint(__uint_as_float(p4[c].x) * (__uint_as_float(v[4 * c + 0]) - Dr)),
                                     __float_as_uint(__uint_as_float(p4[c].y) * (__uint_as_float(v[4 * c + 1]) - Dr)),
                                     __float_as_uint(__uint_as_float(p4[c].z) * (__uint_as_float(v[4 * c + 2]) - Dr)),
                                     __float_as_uint(__uint_as_float(p4[c].w) * (__uint_as_float(v[4 * c + 3]) - Dr)));
          sts128(sb + sw128(lane, c), o);
        }
        fence_proxy_async();
        __syncwarp();
        if (lane == 0) {
          tma_store_2d(&mdb, sb, j0 + q * 32, h * L + i0 + (int)lb);
          tma_store_commit();
          asm volatile("cp.async.bulk.wait_group.read %0;" :: "n"(NSW - 1) : "memory");   // the tile reused next is free
        }
        __syncwarp();
      }
    }
    if (lane == 0) tma_store_wait0();
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 2) { tc_fence_after(); tmem_dealloc(tmem, TCOLS); }
}
