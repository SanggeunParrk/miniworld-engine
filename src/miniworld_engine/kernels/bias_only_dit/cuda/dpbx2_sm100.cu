// dpbx2_sm100.cu -- the bias-only attention's bias gradient (training backward) on CTA PAIRS, sm_100a, bf16 (kind::f16) or -DTF32
// (fp32 operands on kind::tf32: the fp32 path's build of the same kernel):
//
//   dP_h[i, j]  = sum_a sum_d do[a, i, h, d] v[a, j, h, d]          (the attention weights are shared by the A samples)
//   dbias_h     = P_h o (dP_h - D_h),   D_h[i] = sum_a dd[a, h, i]
//
// dpb_sm100.cu is bound by its SMs' TMA intake (doc T3): per sample an item loads a 128-query do tile and an NJ-key v tile for a 128 x NJ
// x DH product (L768, NJ 256: 384 rows per sample for 128 x 256 products; 64 us against a 22 us floor). Here a cluster of two CTAs takes
// a 256-query x 256-key item: each CTA loads its own 128 queries' do tile and HALF of the key tile (128 keys), and the leader issues
// M 256 x N 256 x K 16 products (cta_group::2, B split by N): 256 rows per CTA and sample for the same products -- a third less intake.
// The fp32 twin (dpb32x2_sm100.cu, eng-opt-bias) measured 153 -> 112 us at L768. Used where 256-key tiles divide L (an even number of
// query tiles too) and the pairs' items fill the SM pairs: L768 at L 128-768. Heads as dpb_sm100.cu: two 32-wide heads per item, each its
// own accumulator. TF32 build (as dpb32_sm100.cu / dpb32x2_sm100.cu): a head row is NA = DH / 32 (rounded up) boxes of 32 channels in
// 128-byte swizzle rows (48-wide heads 32 + 16), each box its own 16 KB slot, MMAs of K = 8; P pieces [128][32] fp32 SW128 (16 KB),
// dbias staging [32][32] fp32 SW128 (4 KB). Only the operand format and the stage count change.
//
// ------------------------------------------------------------------------------------------------------------------------- PROTOCOL
// Roles (256 threads): warp 0 lane 0 PRODUCER (both CTAs); warp 1 MMA (leader, rank 0; one elected lane issues); warp 2 TMEM; warp 3
// lane 0 P PRODUCER (each CTA, its own rows of P); warps 4-7 EPILOGUE (a thread = a TMEM lane = a query row). Items li = 0 .. my - 1
// are the same sequence in both CTAs; the K counter g runs over (item, sample).
//  RING stage s = g % NST (this CTA's do tile [128 i][HB d] + its v half [128 j][HB d], SW128; TF32: NBX boxes of each).
//    fill   PRODUCER: g >= NST: wait empty[s] parity ((g / NST) - 1) & 1; leader: expect_tx(full[s], 2 TX); 2 TMA cta_group::2 loads
//           completing on the LEADER's full[s].
//    use    MMA: wait full[s] parity (g / NST) & 1; tcgen05.fence::after_thread_sync; HP (DH / 16) MMAs reading both CTAs' stage s.
//    free   MMA: tcgen05.commit cta_group::2 multicast {0, 1} -> empty[s] (count 1), both CTAs.
//  TMEM (one accumulator per head, HP x 256 columns; ONE set, as dpb_sm100.cu).
//    fill   MMA: li >= 1: wait the leader's acc_empty (acquire.cluster) parity (li - 1) & 1; tcgen05.fence::after_thread_sync; the item's
//           products; after its last sample tcgen05.commit multicast {0, 1} -> acc (count 1), both CTAs.
//    use    EPILOGUE: wait acc parity li & 1; tcgen05.fence::after_thread_sync; tcgen05.ld + wait::ld per 32-key piece.
//    free   each epilogue warp after its last piece of the item: tcgen05.fence::before_thread_sync; __syncwarp; lane 0
//           mbarrier.arrive.release.cluster on the leader's acc_empty (count 8 = 4 warps x 2 CTAs).
//  P ring slot e & 1 ([128 i][32 j]: bf16 SW64 / fp32 SW128), each CTA.
//    fill   P PRODUCER: e >= 2: wait pempty[slot] parity ((e >> 1) - 1) & 1; expect_tx(pfull[slot]); TMA load.
//    use    EPILOGUE: wait pfull[slot] parity (e >> 1) & 1; ld.shared of the row into registers.
//    free   each warp: fence.proxy.async.shared::cta; __syncwarp; lane 0 arrive pempty[slot] (count 4).
//  dbias staging tile t % NSW of a warp ([32 rows][32 j], as P): written by the warp after the TMA store that last read it has
//    read it (lane 0 cp.async.bulk.wait_group.read NSW - 1 after each commit; __syncwarp); fence.proxy.async.shared::cta; __syncwarp;
//    lane 0 TMA store + commit. Exit: lane 0 wait_group 0.
//  No barrier can run two phases ahead of its waiter. Init: thread 0, fence.mbarrier_init, __syncthreads + cluster barrier. Exit:
//  tcgen05.fence::before_thread_sync, __syncthreads, cluster barrier, warp 2 dealloc. No atomics: bit-identical reruns.
// SPDX-License-Identifier: Apache-2.0
#include "sm100.cuh"
using namespace s100;

#ifndef NHEAD
#define NHEAD 16                                             // heads x head width: 16 x 48, 24 x 32, 12 x 64 or 16 x 64
#endif
#ifndef DHEAD
#define DHEAD 48
#endif
constexpr int QM = 128, DH = DHEAD, NH = NHEAD, NJ = 256, NJC = NJ / 2, NQ = NJ / 32;   // NJC: keys per CTA
constexpr int HP = DH == 32 ? 2 : 1, HB = HP * DH;          // heads per item, the item's channels
static_assert(DH == 32 || DH == 48 || DH == 64, "head width 32, 48 or 64");
static_assert(NH % HP == 0 && HP * NJ <= 512, "heads per item, TMEM columns");
#ifdef TF32
constexpr int ESZ = 4, NA = (DH + 31) / 32, WL = DH - 32 * (NA - 1), NBX = HP * NA;   // 32-channel boxes per head (the last WL wide)
constexpr uint32_t IDESC = idesc_tf32(256, NJ);
#else
constexpr int ESZ = 2, NA = 1, WL = DH, NBX = 1;             // the item's HB channels in one 128-byte row
constexpr uint32_t IDESC = idesc_bf16(256, NJ);
#endif
constexpr int TA = QM * 128, TB = NJC * 128;                  // per box: the do tile, this CTA's half of the key tile (16 KB each, SW128)
constexpr int S_DO = 0, S_V = NBX * TA, STB = NBX * (TA + TB);
constexpr int TX = (QM + NJC) * HB * ESZ;                     // bytes one CTA's loads of a stage carry
constexpr int PROW = 32 * ESZ, NPC = PROW / 16;               // a P / dbias row of 32 keys: bytes, 16-byte chunks
constexpr int PC = QM * PROW;                                 // P piece [128 i][32 j]: 8 KB (bf16, SW64) / 16 KB (fp32, SW128)
constexpr int SWB = 32 * PROW, NSW = 3;                       // per-warp dbias staging [32 rows][32 j]
constexpr int O_P = 0, O_S = O_P + 2 * PC, O_ST = O_S + 4 * NSW * SWB + (1024 - (4 * NSW * SWB) % 1024) % 1024;
constexpr int NST = (232448 - O_ST - 1024) / STB;
constexpr int O_BAR = O_ST + NST * STB, SMEM_BYTES = O_BAR + 256;
static_assert(NST >= 2 && SMEM_BYTES <= 232448, "shared memory");
static_assert(O_ST % 1024 == 0 && STB % 1024 == 0, "SW128 stages need 1 KB alignment");
constexpr uint32_t TCOLS = HP * NJ <= 256 ? 256 : 512;
DEVI uint32_t psw(uint32_t r, uint32_t c) { return ESZ == 4 ? sw128(r, c) : sw64(r, c); }   // P / staging rows: SW128 / SW64

struct Bars { uint64_t full[NST], empty[NST], pfull[2], pempty[2], acc, acc_empty; uint32_t tmem; };
static_assert(sizeof(Bars) <= 256, "barriers");

#ifdef TF32
DEVI void umma_pair(uint32_t d, uint64_t a, uint64_t b, uint32_t acc) {
  asm volatile("{ .reg .pred p; setp.ne.b32 p, %4, 0; tcgen05.mma.cta_group::2.kind::tf32 [%0], %1, %2, %3, p; }"
               :: "r"(d), "l"(a), "l"(b), "r"(IDESC), "r"(acc) : "memory");
}
#else
DEVI void umma_pair(uint32_t d, uint64_t a, uint64_t b, uint32_t acc) { umma_ss2(d, a, b, IDESC, acc); }
#endif

// maps: mdo / mdo2 do [A L rows][channels] (bf16: box [HB, 128]; TF32: 32- / 16-channel boxes of 128 rows); mv / mv2 v likewise (any
// row stride); mp P [H L, L] box [32, 128]; mdb dbias [H L, L] box [32, 32] (bf16 SW64, fp32 SW128). dd [A, NH, L] fp32.
// The kernel's body on a virtual grid (CTA vb of vg, vb even on a cluster's leader): bo_dpbx2_sm100 below runs it on the real one,
// bo_pvdpb.cu next to pv dV in one launch (-DBODY_ONLY leaves the kernel out). The maps are the kernel's __grid_constant__ parameters.
DEVI void dpbx2_body(const CUtensorMap& mdo, const CUtensorMap& mdo2, const CUtensorMap& mv, const CUtensorMap& mv2,
                     const CUtensorMap& mp, const CUtensorMap& mdb, const float* __restrict__ dd, int L, int A, const int vb, const int vg) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  Bars& B = *reinterpret_cast<Bars*>(sm + O_BAR);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int crank = (int)cluster_rank();
  const bool leader = crank == 0;
  const int mp2 = L / (2 * QM), nj = L / NJ, items = (NH / HP) * mp2 * nj;   // the host serves L % 256 == 0 only
  const int npairs = vg >> 1, pair = vb >> 1;
  const int my = (items > pair) ? (items - pair + npairs - 1) / npairs : 0;
  auto item_of = [&](int li, int& h, int& i0, int& j0) {      // the item's first head, THIS CTA's query tile, the key tile
    const int wi = pair + li * npairs;
    j0 = (wi % nj) * NJ;
    const int r = wi / nj;
    i0 = (2 * (r % mp2) + crank) * QM;
    h = (r / mp2) * HP;
  };

  if (tid == 0) {
    for (int s = 0; s < NST; ++s) { mbar_init(&B.full[s], 1); mbar_init(&B.empty[s], 1); }
    for (int e = 0; e < 2; ++e) { mbar_init(&B.pfull[e], 1); mbar_init(&B.pempty[e], 4); }
    mbar_init(&B.acc, 1); mbar_init(&B.acc_empty, 8);
    fence_barrier_init();
    prefetch_map(&mdo); prefetch_map(&mv); prefetch_map(&mp); prefetch_map(&mdb);
    if (WL == 16) { prefetch_map(&mdo2); prefetch_map(&mv2); }
  }
  if (warp == 2) { tmem_alloc2(smem_u32(&B.tmem), TCOLS); tmem_relinquish2(); }
  tc_fence_before();
  __syncthreads();
  cluster_sync();
  tc_fence_after();
  const uint32_t tmem = B.tmem;

  if (warp == 0) {
    // ---------------------------------------------------------------------------------------------------------- PRODUCER
    if (lane == 0) {
      int g = 0;
      for (int li = 0; li < my; ++li) {
        int h, i0, j0; item_of(li, h, i0, j0);
        for (int a = 0; a < A; ++a, ++g) {
          const int s = g % NST;
          if (g >= NST) mbar_wait(&B.empty[s], ((g / NST) - 1) & 1);
          if (leader) mbar_expect_tx(&B.full[s], 2 * TX);
          const uint32_t st = su + O_ST + s * STB;
#pragma unroll
          for (int hh = 0; hh < (NBX > 1 ? HP : 1); ++hh)
#pragma unroll
            for (int c = 0; c < NA; ++c) {
              const int bx = hh * NA + c, col = (h + hh) * DH + 32 * c;      // bf16: one box, the item's HB channels
              const bool narrow = c == NA - 1 && WL == 16 && NBX > 1;
              tma_load_2d_2sm(st + S_DO + bx * TA, narrow ? &mdo2 : &mdo, &B.full[s], col, a * L + i0);
              tma_load_2d_2sm(st + S_V + bx * TB, narrow ? &mv2 : &mv, &B.full[s], col, a * L + j0 + crank * NJC);
            }
        }
      }
    }
  } else if (warp == 1) {
    // ---------------------------------------------------------------------------------------------------------- MMA (leader)
    if (leader) {
      int g = 0;
      for (int li = 0; li < my; ++li) {
        if (li >= 1) mbar_wait_cl(&B.acc_empty, (li - 1) & 1);
        tc_fence_after();
        for (int a = 0; a < A; ++a, ++g) {
          const int s = g % NST;
          mbar_wait(&B.full[s], (g / NST) & 1);
          tc_fence_after();
          const uint32_t st = su + O_ST + s * STB;
          if (elect_one()) {
#ifdef TF32
#pragma unroll
            for (int hh = 0; hh < HP; ++hh)
#pragma unroll
              for (int c = 0; c < NA; ++c) {
                const int bx = hh * NA + c;
                const uint64_t da = desc_k128(st + S_DO + bx * TA), dv = desc_k128(st + S_V + bx * TB);
#pragma unroll
                for (int ks = 0; ks < (c == NA - 1 ? WL : 32) / 8; ++ks)       // K = 8 fp32 = 32 B per MMA
                  umma_pair(tmem + hh * NJ, da + (uint64_t)(ks * 2), dv + (uint64_t)(ks * 2), (a > 0 || c > 0 || ks > 0) ? 1u : 0u);
              }
#else
            const uint64_t da = desc_k128(st + S_DO), dv = desc_k128(st + S_V);
#pragma unroll
            for (int hh = 0; hh < HP; ++hh)
#pragma unroll
              for (int ks = 0; ks < DH / 16; ++ks) {
                const uint64_t ko = (uint64_t)(hh * (DH * 2 >> 4) + ks * 2);   // head hh's columns, K step ks (16-byte units)
                umma_pair(tmem + hh * NJ, da + ko, dv + ko, (a > 0 || ks > 0) ? 1u : 0u);
              }
#endif
            tc_commit2_mc(&B.empty[s], 3);
            if (a == A - 1) tc_commit2_mc(&B.acc, 3);
          }
          __syncwarp();
        }
      }
    }
  } else if (warp == 3) {
    // ---------------------------------------------------------------------------------------------------------- P PRODUCER
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
    // ---------------------------------------------------------------------------------------------------------- EPILOGUE
    const uint32_t lb = (uint32_t)(warp & 3) * 32, r = lb + lane, trow = tmem + (lb << 16);
    const uint32_t stg = su + O_S + (uint32_t)(warp & 3) * NSW * SWB;
    int e = 0, t = 0;
    for (int li = 0; li < my; ++li) {
      int h, i0, j0; item_of(li, h, i0, j0);
      float Drs[HP];                                            // D[i], read while the item's products run
#pragma unroll
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
          tmem_ld32(trow + hh * NJ + q * 32, v);               // keys j0 + 32 q ..: columns 0-127 the leader's half, 128-255 the peer's
          const int slot = e & 1;
          mbar_wait(&B.pfull[slot], (e >> 1) & 1);
          uint4 p4[NPC];
#pragma unroll
          for (int c = 0; c < NPC; ++c) p4[c] = lds128(su + O_P + slot * PC + psw(r, c));
          fence_proxy_async();                                  // these reads before the P producer's async refill
          __syncwarp();
          if (lane == 0) mbar_arrive(&B.pempty[slot]);
          tmem_wait_ld();
          if (hh == HP - 1 && q == NQ - 1) {                    // the accumulator: this warp's last read of the item
            tc_fence_before();
            __syncwarp();
            if (lane == 0) mbar_arrive_remote(&B.acc_empty, 0);
          }
          uint4 o4[NPC];
#pragma unroll
          for (int c = 0; c < NPC; ++c) {
            const uint32_t pw[4] = {p4[c].x, p4[c].y, p4[c].z, p4[c].w};
            uint32_t o[4];
#pragma unroll
            for (int k = 0; k < 4; ++k)
#ifdef TF32
              o[k] = __float_as_uint(__uint_as_float(pw[k]) * (__uint_as_float(v[4 * c + k]) - Dr));
#else
              o[k] = pack_bf16(bf16lo(pw[k]) * (__uint_as_float(v[8 * c + 2 * k]) - Dr),
                               bf16hi(pw[k]) * (__uint_as_float(v[8 * c + 2 * k + 1]) - Dr));
#endif
            o4[c] = make_uint4(o[0], o[1], o[2], o[3]);
          }
          const uint32_t sb = stg + (uint32_t)(t % NSW) * SWB;  // free: the store that read it last (t - NSW) has read it
#pragma unroll
          for (int c = 0; c < NPC; ++c) sts128(sb + psw(lane, c), o4[c]);
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
    }
    if (lane == 0) tma_store_wait0();
  }
  tc_fence_before();
  __syncthreads();
  cluster_sync();
  if (warp == 2) { tc_fence_after(); tmem_dealloc2(tmem, TCOLS); }
}

#ifndef BODY_ONLY
extern "C" __global__ void __launch_bounds__(256, 1)
bo_dpbx2_sm100(const __grid_constant__ CUtensorMap mdo, const __grid_constant__ CUtensorMap mdo2,
               const __grid_constant__ CUtensorMap mv, const __grid_constant__ CUtensorMap mv2,
               const __grid_constant__ CUtensorMap mp, const __grid_constant__ CUtensorMap mdb,
               const float* __restrict__ dd, int L, int A) {
  dpbx2_body(mdo, mdo2, mv, mv2, mp, mdb, dd, L, A, (int)blockIdx.x, (int)gridDim.x);
}
#endif
