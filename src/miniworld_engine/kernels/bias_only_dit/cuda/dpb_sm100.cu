// dpb_sm100.cu -- the bias-only attention's bias gradient, sm_100a (training backward):
//
//   dP_h[i, j]  = sum_a sum_d do[a, i, h, d] v[a, j, h, d]          (the attention weights are shared by the A samples)
//   dbias_h     = P_h o (dP_h - D_h),   D_h[i] = sum_a sum_d do[a, i, h, d] o[a, i, h, d]   (given, per sample: dd [A, 16, L])
//
// do, v: [A L, *] bf16 rows (a, token), head h in columns DH h .. DH h + DH - 1 (v is the v half of the v|g GEMM output, any row
// stride); P, dbias: [16 L, L] bf16 (head-major, query rows). A work item is (head, 128-query tile, NJ-key tile); its K loop walks
// the samples: per sample one do tile [128 i][DH d] (K-major) and the key tile [NJ j][DH d] in two NJ / 2 halves, products
// M 128 x N NJ / 2 x K DH into an NJ-column TMEM accumulator (DH = 48, 32 or 64: -DNHEAD / -DDHEAD, 16 x 48, 24 x 32 or 12 x 64 heads). The epilogue streams P in [128][32] pieces (SW64) through a
// two-slot ring loaded by its own producer warp, and each warp stores its 32 rows of dbias per piece.
// Warps: 0 K-loop TMA producer, 1 MMA, 2 TMEM allocator, 3 P producer, 4-7 epilogue (one query row per thread).
// -DPSB (a pair bias of its own per sample): P, dbias [H, A, L, L] (a plane per (head, sample)), nothing summed over the samples:
//   dbias_{h,a} = P_{h,a} o (do_a v_a^T - D_{h,a}),  D_{h,a}[i] = dd[a, h, i]
// a work item is (sample, head group, query tile, key tile) and its K loop the one sample's do / v. The key mask (``km``: kmode 1 one
// [L] for all samples, 2 one [A, L] per sample) zeroes the masked keys' dbias, as the module's masked_fill does: a sample whose keys
// are ALL masked has uniform P rows (the module's finfo.min fill), so P alone would not zero them.
// SPDX-License-Identifier: Apache-2.0
#include "sm100.cuh"
using namespace s100;

#ifndef NJ
#define NJ 384
#endif
#ifndef NHEAD
#define NHEAD 16                                             // heads x head width: 16 x 48, 24 x 32 or 12 x 64
#endif
#ifndef DHEAD
#define DHEAD 48
#endif
constexpr int QM = 128, DH = DHEAD, NH = NHEAD, NJ2 = NJ / 2, NQ = NJ / 32;
#ifdef PSB
constexpr bool PSB_P = true;
#else
constexpr bool PSB_P = false;
#endif
// heads per work item: two 32-wide heads fill one 128-byte row (the do / v boxes 16 KB instead of 8, half the TMA instructions;
// the SM's TMA intake rises with the box size), each head its own accumulator, read from its half of the row
constexpr int HP = DH == 32 ? 2 : 1, HB = HP * DH;
static_assert(DH % 16 == 0 && HB <= 64, "head width: a multiple of 16, the item's heads within one 128-byte swizzle row");
static_assert(NH % HP == 0 && HP * NJ <= 512, "heads per item, TMEM columns");
static_assert(NJ2 % 16 == 0 && NJ2 <= 256 && NJ % 32 == 0, "key tile");
constexpr int TA = QM * 128, TBH = NJ2 * 128, STB = TA + 2 * TBH;
constexpr int PC = QM * 64;                                  // P piece [128 i][32 j] bf16 SW64: 8 KB
constexpr int SWB = 32 * 64, NSW = 3;                        // per-warp dbias staging [32 rows][32 cols] bf16 SW64
constexpr int O_P = 0, O_S = O_P + 2 * PC, O_ST = O_S + 4 * NSW * SWB + (1024 - (4 * NSW * SWB) % 1024) % 1024;
constexpr int NST = (232448 - O_ST - 1024) / STB;
constexpr int O_BAR = O_ST + NST * STB, SMEM_BYTES = O_BAR + 256;
static_assert(NST >= 2 && SMEM_BYTES <= 232448, "shared memory");
static_assert(O_ST % 1024 == 0, "SW128 stages need 1 KB alignment");
constexpr uint32_t IDESC = idesc_bf16(QM, NJ2), TCOLS = HP * NJ <= 256 ? 256 : 512;

struct Bars { uint64_t full[NST], empty[NST], pfull[2], pempty[2], acc, acc_empty; uint32_t tmem; };

// The kernel's body on a virtual grid (CTA vb of vg): bo_dpb_sm100 below runs it on the real one, bo_pvdpb.cu next to pv dV in one
// launch (-DBODY_ONLY leaves the kernel out). The maps are the kernel's __grid_constant__ parameters.
DEVI void dpb_body(const CUtensorMap& mdo, const CUtensorMap& mv, const CUtensorMap& mp, const CUtensorMap& mdb,
                   const float* __restrict__ dd, int L, int A, const int vb, const int vg, const bool* __restrict__ km = nullptr,
                   int kmode = 0) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  Bars& B = *reinterpret_cast<Bars*>(sm + O_BAR);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  // PSB: items over (sample, head group, query tile, key tile), the K loop one sample; else the K loop walks the A samples
  const int nsi = PSB_P ? A : 1, nk = PSB_P ? 1 : A;
  const int mt = L / QM, nj = L / NJ, items = nsi * (NH / HP) * mt * nj;
  const int my = (items > vb) ? (items - vb + vg - 1) / vg : 0;
  auto item_of = [&](int li, int& h, int& i0, int& j0, int& a0) {
    const int wi = vb + li * vg;
    j0 = (wi % nj) * NJ;
    const int r = wi / nj;
    i0 = (r % mt) * QM;
    const int r2 = r / mt;
    h = (r2 % (NH / HP)) * HP;                                   // the item's first head
    a0 = r2 / (NH / HP);                                         // PSB: the item's sample (else 0)
  };
  auto plane = [&](int hd, int a0) { return (PSB_P ? hd * A + a0 : hd) * L; };   // first row of a P / dbias plane

  if (tid == 0) {
    for (int s = 0; s < NST; ++s) { mbar_init(&B.full[s], 1); mbar_init(&B.empty[s], 1); }
    for (int e = 0; e < 2; ++e) { mbar_init(&B.pfull[e], 1); mbar_init(&B.pempty[e], 4); }
    mbar_init(&B.acc, 1); mbar_init(&B.acc_empty, 4);
    fence_barrier_init();
    prefetch_map(&mdo); prefetch_map(&mv); prefetch_map(&mp); prefetch_map(&mdb);
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
        int h, i0, j0, a0; item_of(li, h, i0, j0, a0);
        for (int k = 0; k < nk; ++k, ++g) {
          const int s = g % NST, a = a0 + k;
          if (g >= NST) mbar_wait(&B.empty[s], ((g / NST) - 1) & 1);
          const uint32_t st = su + O_ST + s * STB;
          mbar_expect_tx(&B.full[s], QM * HB * 2 + 2 * NJ2 * HB * 2);
          tma_load_2d(st, &mdo, &B.full[s], h * DH, a * L + i0);
          tma_load_2d(st + TA, &mv, &B.full[s], h * DH, a * L + j0);
          tma_load_2d(st + TA + TBH, &mv, &B.full[s], h * DH, a * L + j0 + NJ2);
        }
      }
    }
  } else if (warp == 1) {
    int g = 0;
    for (int li = 0; li < my; ++li) {
      if (li >= 1) mbar_wait(&B.acc_empty, (li - 1) & 1);
      tc_fence_after();
      for (int a = 0; a < nk; ++a, ++g) {
        const int s = g % NST;
        mbar_wait(&B.full[s], (g / NST) & 1);
        tc_fence_after();
        const uint32_t st = su + O_ST + s * STB;
        const uint64_t da = desc_k128(st), d0 = desc_k128(st + TA), d1 = desc_k128(st + TA + TBH);
        if (elect_one()) {
#pragma unroll
          for (int hh = 0; hh < HP; ++hh)
#pragma unroll
            for (int ks = 0; ks < DH / 16; ++ks) {
              const uint32_t accf = (a > 0 || ks > 0) ? 1u : 0u;
              const uint64_t ko = (uint64_t)(hh * (DH * 2 >> 4) + ks * 2);       // head hh's columns, K step ks (16 bytes units)
              umma_ss(tmem + hh * NJ, da + ko, d0 + ko, IDESC, accf);
              umma_ss(tmem + hh * NJ + NJ2, da + ko, d1 + ko, IDESC, accf);
            }
          tc_commit(&B.empty[s]);
          if (a == nk - 1) tc_commit(&B.acc);
        }
        __syncwarp();
      }
    }
  } else if (warp == 3) {
    if (lane == 0) {
      int e = 0;
      for (int li = 0; li < my; ++li) {
        int h, i0, j0, a0; item_of(li, h, i0, j0, a0);
        for (int hh = 0; hh < HP; ++hh)
          for (int q = 0; q < NQ; ++q, ++e) {
            const int slot = e & 1;
            if (e >= 2) mbar_wait(&B.pempty[slot], ((e >> 1) - 1) & 1);
            mbar_expect_tx(&B.pfull[slot], QM * 32 * 2);
            tma_load_2d(su + O_P + slot * PC, &mp, &B.pfull[slot], j0 + q * 32, plane(h + hh, a0) + i0);
          }
      }
    }
  } else if (warp >= 4) {
    const uint32_t lb = (uint32_t)(warp & 3) * 32, r = lb + lane, trow = tmem + (lb << 16);
    const uint32_t stg = su + O_S + (uint32_t)(warp & 3) * NSW * SWB;
    int e = 0, t = 0;
    for (int li = 0; li < my; ++li) {
      int h, i0, j0, a0; item_of(li, h, i0, j0, a0);
      float Drs[HP];                                                // D[i] = sum over the item's samples of the per-sample row sums,
#pragma unroll                                                      // read while the item's products run
      for (int hh = 0; hh < HP; ++hh) {
        Drs[hh] = 0.f;
        for (int a = a0; a < a0 + nk; ++a) Drs[hh] += __ldg(dd + ((long)a * NH + h + hh) * L + i0 + r);
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
          unsigned char mk[32];                                     // PSB: the piece's 32 keys kept (1) or masked (0)
          if (PSB_P && kmode) {
            const bool* mrow = km + (kmode == 2 ? (long)a0 * L : 0L) + j0 + q * 32;
            *reinterpret_cast<uint4*>(mk) = __ldg(reinterpret_cast<const uint4*>(mrow));
            *reinterpret_cast<uint4*>(mk + 16) = __ldg(reinterpret_cast<const uint4*>(mrow + 16));
          }
          const int slot = e & 1;
          mbar_wait(&B.pfull[slot], (e >> 1) & 1);
          uint4 p4[4];
#pragma unroll
          for (int c = 0; c < 4; ++c) p4[c] = lds128(su + O_P + slot * PC + sw64(r, c));
          __syncwarp();
          if (lane == 0) mbar_arrive(&B.pempty[slot]);
          tmem_wait_ld();
          uint4 o4[4];
#pragma unroll
          for (int c = 0; c < 4; ++c) {
            const uint32_t pw[4] = {p4[c].x, p4[c].y, p4[c].z, p4[c].w};
            uint32_t o[4];
#pragma unroll
            for (int k = 0; k < 4; ++k) {
              float lo = bf16lo(pw[k]) * (__uint_as_float(v[8 * c + 2 * k]) - Dr);
              float hi = bf16hi(pw[k]) * (__uint_as_float(v[8 * c + 2 * k + 1]) - Dr);
              if (PSB_P && kmode) { lo = mk[8 * c + 2 * k] ? lo : 0.f; hi = mk[8 * c + 2 * k + 1] ? hi : 0.f; }
              o[k] = pack_bf16(lo, hi);
            }
            o4[c] = make_uint4(o[0], o[1], o[2], o[3]);
          }
          const uint32_t sb = stg + (uint32_t)(t % NSW) * SWB;
#pragma unroll
          for (int c = 0; c < 4; ++c) sts128(sb + sw64(lane, c), o4[c]);
          fence_proxy_async();
          __syncwarp();
          if (lane == 0) {
            tma_store_2d(&mdb, sb, j0 + q * 32, plane(h + hh, a0) + i0 + (int)lb);
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

#ifndef BODY_ONLY
#ifdef PSB
extern "C" __global__ void __launch_bounds__(256, 1)
bo_dpb_sm100(const __grid_constant__ CUtensorMap mdo, const __grid_constant__ CUtensorMap mv,
             const __grid_constant__ CUtensorMap mp, const __grid_constant__ CUtensorMap mdb,
             const float* __restrict__ dd, int L, int A, const bool* __restrict__ km, int kmode) {
  dpb_body(mdo, mv, mp, mdb, dd, L, A, (int)blockIdx.x, (int)gridDim.x, km, kmode);
}
#else
extern "C" __global__ void __launch_bounds__(256, 1)
bo_dpb_sm100(const __grid_constant__ CUtensorMap mdo, const __grid_constant__ CUtensorMap mv,
             const __grid_constant__ CUtensorMap mp, const __grid_constant__ CUtensorMap mdb,
             const float* __restrict__ dd, int L, int A) {
  dpb_body(mdo, mv, mp, mdb, dd, L, A, (int)blockIdx.x, (int)gridDim.x);
}
#endif
#endif
