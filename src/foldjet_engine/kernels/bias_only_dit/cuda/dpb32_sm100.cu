// dpb32_sm100.cu -- the bias-only attention's bias gradient for the fp32 path, sm_100a, TF32 tensor cores (training backward): the
// fp32 port of dpb_sm100.cu, its structure unchanged -- work items, warp roles, the K-loop and P rings, ONE accumulator released after
// the item, per-warp dbias staging --
//
//   dP_h[i, j]  = sum_a sum_d do[a, i, h, d] v[a, j, h, d]          (the attention weights are shared by the A samples)
//   dbias_h     = P_h o (dP_h - D_h),   D_h[i] = sum_a dd[a, h, i]   (dd [A, H, L] fp32: per sample sum_d do o)
//
// do, v: [A L, *] fp32 rows (a, token), head h in columns DH h .. DH h + DH - 1 (v the v half of the v|g GEMM output, any row
// stride); P, dbias: [H L, L] fp32 (head-major, query rows). A work item is (head group, 128-query tile, NJ-key tile); its K loop
// walks the samples: per sample the do tile [128 i][DH d] (K-major) and the key tile [NJ j][DH d] in two NJ / 2 halves, products
// M 128 x N NJ / 2 x K DH (kind::tf32, K = 8 per MMA: fp32 operands, the MMA keeps their top 19 bits; fp32 accumulation) into an
// NJ-column TMEM accumulator per head. The epilogue streams P in [128][32] pieces through a two-slot ring loaded by its own producer
// warp, and each warp stores its 32 rows of dbias per piece.
//
// What fp32 changes against dpb_sm100.cu: a head row is DH x 4 bytes, so it is NA = DH / 32 (rounded up) boxes of 32 channels in
// 128-byte swizzle rows (48-wide heads: 32 + 16 channels, the 16-channel box in the first 64 bytes of its rows), 4 or 2 K steps each;
// the P piece is 16 KB (SW128) and the dbias staging tile 4 KB (SW128); and the key tile is NJ = 128 (two 64-key halves): a stage is
// HP NA (16 KB + 16 KB) = 32 KB, so two stages fit beside the P ring and the staging (bf16: NJ up to 384). With NJ = 128 the key
// tile is ONE product of N = NJ per K step (bf16 splits its up to 384 keys into two N <= 192 halves): the do tile is read from shared
// memory once per K step, not twice, and half the MMAs are issued (the two-half build ran at 38 % of the MMA floor, L768 159 us);
// the key tile arrives as one [NJ][32] box.
// Warps: 0 K-loop TMA producer, 1 MMA (whole warp waits, elect_one() issues), 2 TMEM allocator, 3 P producer, 4-7 epilogue (one
// query row per thread: 32 accumulator registers + 32 P values per piece; at most 128 registers).
// SPDX-License-Identifier: Apache-2.0
#include "sm100.cuh"
using namespace s100;

#ifndef NJ
#define NJ 128
#endif
#ifndef NHEAD
#define NHEAD 16                                             // heads x head width: 16 x 48, 24 x 32, 12 x 64 or 16 x 64
#endif
#ifndef DHEAD
#define DHEAD 48
#endif
constexpr int QM = 128, DH = DHEAD, NH = NHEAD, NJ2 = NJ / 2, NQ = NJ / 32;
// heads per work item (as dpb_sm100.cu: two 32-wide heads per item), each its own accumulator
constexpr int HP = DH == 32 ? 2 : 1;
constexpr int NA = (DH + 31) / 32;                           // 32-channel boxes of a head row (the last one DH - 32 (NA - 1) wide)
constexpr int WL = DH - 32 * (NA - 1);                       // channels of the last box: 32 or 16
constexpr int NBX = HP * NA;                                 // boxes per tile row
static_assert(DH == 32 || DH == 48 || DH == 64, "head width 32, 48 or 64");
static_assert(NH % HP == 0 && HP * NJ <= 512, "heads per item, TMEM columns");
static_assert(NJ2 % 16 == 0 && NJ2 <= 256 && NJ % 32 == 0, "key tile");
constexpr int TA = QM * 128, TBH = NJ2 * 128, TB = 2 * TBH;  // per box: the do tile, a key-tile half, the key tile
constexpr int S_DO = 0, S_V = NBX * TA, STB = NBX * (TA + TB);
constexpr int TX = HP * DH * 4 * (QM + NJ);                  // bytes a stage receives
constexpr int PC = QM * 128;                                 // P piece [128 i][32 j] fp32 SW128: 16 KB
constexpr int SWB = 32 * 128, NSW = 3;                       // per-warp dbias staging [32 rows][32 cols] fp32 SW128
constexpr int O_P = 0, O_S = O_P + 2 * PC, O_ST = O_S + 4 * NSW * SWB;
constexpr int NST = (232448 - O_ST - 1024) / STB;
constexpr int O_BAR = O_ST + NST * STB, SMEM_BYTES = O_BAR + 256;
static_assert(NST >= 2 && SMEM_BYTES <= 232448, "shared memory");
static_assert(O_ST % 1024 == 0 && STB % 1024 == 0 && TBH % 1024 == 0, "SW128 tiles need 1 KB alignment");
static_assert(NJ <= 256, "the key tile is one N = NJ product");
constexpr uint32_t IDESC = idesc_tf32(QM, NJ), TCOLS = HP * NJ <= 256 ? 256 : 512;

struct Bars { uint64_t full[NST], empty[NST], pfull[2], pempty[2], acc, acc_empty; uint32_t tmem; };

// maps: mdo / mv: 32-channel boxes [32][128] / [32][NJ / 2]; mdo2 / mv2: the 16-channel boxes of 48-wide heads (else unused)
extern "C" __global__ void __launch_bounds__(256, 2)
bo_dpb32_sm100(const __grid_constant__ CUtensorMap mdo, const __grid_constant__ CUtensorMap mdo2,
               const __grid_constant__ CUtensorMap mv, const __grid_constant__ CUtensorMap mv2,
               const __grid_constant__ CUtensorMap mp, const __grid_constant__ CUtensorMap mdb,
               const float* __restrict__ dd, int L, int A) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  Bars& B = *reinterpret_cast<Bars*>(sm + O_BAR);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int mt = L / QM, nj = L / NJ, items = (NH / HP) * mt * nj;
  const int my = (items > (int)blockIdx.x) ? (items - (int)blockIdx.x + (int)gridDim.x - 1) / (int)gridDim.x : 0;
  auto item_of = [&](int li, int& h, int& i0, int& j0) {
    const int wi = (int)blockIdx.x + li * (int)gridDim.x;
    j0 = (wi % nj) * NJ;
    const int r = wi / nj;
    i0 = (r % mt) * QM; h = (r / mt) * HP;                       // the item's first head
  };

  if (tid == 0) {
    for (int s = 0; s < NST; ++s) { mbar_init(&B.full[s], 1); mbar_init(&B.empty[s], 1); }
    for (int e = 0; e < 2; ++e) { mbar_init(&B.pfull[e], 1); mbar_init(&B.pempty[e], 4); }
    mbar_init(&B.acc, 1); mbar_init(&B.acc_empty, 4);
    fence_barrier_init();
    prefetch_map(&mdo); prefetch_map(&mv); prefetch_map(&mp); prefetch_map(&mdb);
    if (WL == 16) { prefetch_map(&mdo2); prefetch_map(&mv2); }
  }
  if (warp == 2) { tmem_alloc(smem_u32(&B.tmem), TCOLS); tmem_relinquish(); }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = B.tmem;

  if (warp == 0) {
    if (lane == 0) {
      int g = 0;
      for (int li = 0; li < my; ++li) {
        int h, i0, j0; item_of(li, h, i0, j0);
        for (int a = 0; a < A; ++a, ++g) {
          const int s = g % NST;
          if (g >= NST) mbar_wait(&B.empty[s], ((g / NST) - 1) & 1);
          const uint32_t st = su + O_ST + s * STB;
          mbar_expect_tx(&B.full[s], TX);
#pragma unroll
          for (int hh = 0; hh < HP; ++hh)
#pragma unroll
            for (int c = 0; c < NA; ++c) {
              const int bx = hh * NA + c, col = (h + hh) * DH + 32 * c;
              const CUtensorMap* md = (c == NA - 1 && WL == 16) ? &mdo2 : &mdo;
              const CUtensorMap* mk = (c == NA - 1 && WL == 16) ? &mv2 : &mv;
              tma_load_2d(st + S_DO + bx * TA, md, &B.full[s], col, a * L + i0);
              tma_load_2d(st + S_V + bx * TB, mk, &B.full[s], col, a * L + j0);
            }
        }
      }
    }
  } else if (warp == 1) {
    int g = 0;
    for (int li = 0; li < my; ++li) {
      if (li >= 1) mbar_wait(&B.acc_empty, (li - 1) & 1);
      tc_fence_after();
      for (int a = 0; a < A; ++a, ++g) {
        const int s = g % NST;
        mbar_wait(&B.full[s], (g / NST) & 1);
        tc_fence_after();
        const uint32_t st = su + O_ST + s * STB;
        if (elect_one()) {
#pragma unroll
          for (int hh = 0; hh < HP; ++hh)
#pragma unroll
            for (int c = 0; c < NA; ++c) {
              const int bx = hh * NA + c;
              const uint64_t da = desc_k128(st + S_DO + bx * TA), dk = desc_k128(st + S_V + bx * TB);
#pragma unroll
              for (int ks = 0; ks < (c == NA - 1 ? WL : 32) / 8; ++ks) {      // K = 8 fp32 = 32 B per MMA
                const uint32_t accf = (a > 0 || c > 0 || ks > 0) ? 1u : 0u;
                umma_ss_tf32(tmem + hh * NJ, da + (uint64_t)(ks * 2), dk + (uint64_t)(ks * 2), IDESC, accf);
              }
            }
          tc_commit(&B.empty[s]);
          if (a == A - 1) tc_commit(&B.acc);
        }
        __syncwarp();
      }
    }
  } else if (warp == 3) {
    if (lane == 0) {
      int e = 0;
      for (int li = 0; li < my; ++li) {
        int h, i0, j0; item_of(li, h, i0, j0);
        for (int hh = 0; hh < HP; ++hh)
          for (int q = 0; q < NQ; ++q, ++e) {
            const int slot = e & 1;
            if (e >= 2) mbar_wait(&B.pempty[slot], ((e >> 1) - 1) & 1);
            mbar_expect_tx(&B.pfull[slot], PC);
            tma_load_2d(su + O_P + slot * PC, &mp, &B.pfull[slot], j0 + q * 32, (h + hh) * L + i0);
          }
      }
    }
  } else if (warp >= 4) {
    const uint32_t lb = (uint32_t)(warp & 3) * 32, r = lb + lane, trow = tmem + (lb << 16);
    const uint32_t stg = su + O_S + (uint32_t)(warp & 3) * NSW * SWB;
    int e = 0, t = 0;
    for (int li = 0; li < my; ++li) {
      int h, i0, j0; item_of(li, h, i0, j0);
      float Drs[HP];                                                // D[i] = sum over the samples of the per-sample row sums,
#pragma unroll                                                      // read while the item's products run
      for (int hh = 0; hh < HP; ++hh) {
        Drs[hh] = 0.f;
        for (int a = 0; a < A; ++a) Drs[hh] += __ldg(dd + ((long)a * NH + h + hh) * L + i0 + r);
      }
      mbar_wait(&B.acc, li & 1);
      tc_fence_after();
#pragma unroll
      for (int hh = 0; hh < HP; ++hh) {
        const float Dr = Drs[hh];
#pragma unroll 1
        for (int q = 0; q < NQ; ++q, ++e, ++t) {
          uint32_t v[32];
          tmem_ld32(trow + hh * NJ + q * 32, v);
          const int slot = e & 1;
          mbar_wait(&B.pfull[slot], (e >> 1) & 1);
          uint4 p4[8];
#pragma unroll
          for (int c = 0; c < 8; ++c) p4[c] = lds128(su + O_P + slot * PC + sw128(r, c));
          __syncwarp();
          if (lane == 0) mbar_arrive(&B.pempty[slot]);
          tmem_wait_ld();
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
            tma_store_2d(&mdb, sb, j0 + q * 32, (h + hh) * L + i0 + (int)lb);
            tma_store_commit();
            asm volatile("cp.async.bulk.wait_group.read %0;" :: "n"(NSW - 1) : "memory");
          }
          __syncwarp();
        }
      }
      tc_fence_before();
      __syncwarp();
      if (lane == 0) mbar_arrive(&B.acc_empty);
    }
    if (lane == 0) tma_store_wait0();
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 2) { tc_fence_after(); tmem_dealloc(tmem, TCOLS); }
}
