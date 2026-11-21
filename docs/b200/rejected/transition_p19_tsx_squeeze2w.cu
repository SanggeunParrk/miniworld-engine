// tsx_squeeze2w.cu -- the small-M Transition squeeze out = h Ws^T + x on a CTA PAIR with M = 256 products (tsx_squeeze2.cu's M = 128
// pair products run the tensor cores at half rate: 64 rows a CTA), sm_100a, bf16, inference (-DDIM=<D> -DHID=<H> -DNBW=<output
// columns per pair: 64 / 96 / 128> -DKBX=<K-blocks per ring stage>).
// One 2-CTA cluster per (256-row block rb, NBW-column block): CTA r holds the 128 rows of tile 2 rb + r of h and Ws rows
// [c0 + r NBW / 2, c0 + (r + 1) NBW / 2); the leader issues M = 256, N = NBW products over the whole hidden width and each CTA receives
// its 128 rows x NBW columns (0 .. NBW / 2 the leader's Ws rows, NBW / 2 .. NBW the peer's). The Ws K-blocks of the first ring round are
// loaded before griddepcontrol.wait (parameters), lane s of the producer warp issuing ring stage s. Rows past m: zero-filled by TMA,
// not stored.
// SPDX-License-Identifier: Apache-2.0
#include "sm100.cuh"
#include "tsx_store.cuh"
using namespace s100;

#ifndef DIM
#define DIM 768
#endif
#ifndef HID
#define HID (2 * DIM)
#endif
#ifndef NBW
#define NBW 96
#endif
#ifndef KBX
#define KBX 2
#endif
constexpr int D_ = DIM, H_ = HID, ROWS = 128, NB = NBW, NBH = NB / 2, NBLK = D_ / NB, NKB = H_ / 64;
static_assert(NB % 32 == 0 && NB >= 64 && NB <= 256 && D_ % NB == 0 && H_ % 64 == 0, "32-column epilogue passes, whole K-blocks");
constexpr int AKB = ROWS * 128, WKB = NBH * 128;                // one K-block of this CTA's 128 h rows (16 KB) / of its NB / 2 Ws rows
constexpr int NS = NKB / KBX, AST = KBX * AKB, WST = KBX * WKB;
static_assert(NKB % KBX == 0, "whole stages");
constexpr int CAP = (232448 - 1024 - 512) / (AST + WST);
constexpr int STG = (NS < CAP ? NS : CAP) < 32 ? (NS < CAP ? NS : CAP) : 32;
static_assert(STG >= 2, "ring stages");
constexpr int O_A = 0, O_W = STG * AST, O_BAR = O_W + STG * WST, SMEM_BYTES = O_BAR + 512;
static_assert(SMEM_BYTES <= 232448, "shared memory budget");
constexpr uint32_t TCOLS = NB <= 64 ? 64 : NB <= 128 ? 128 : 256;
constexpr uint32_t IDESC = idesc_bf16(256, NB);
typedef __nv_bfloat16 ACT;

DEVI void tma_load_3d_2sm(uint32_t dst, const CUtensorMap* m, uint64_t* bar, int c0, int c1, int c2) {
  asm volatile("cp.async.bulk.tensor.3d.cta_group::2.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5}], [%2];"
               :: "r"(dst), "l"(m), "r"(smem_u32(bar) & 0xFEFFFFFFu), "r"(c0), "r"(c1), "r"(c2) : "memory");
}

struct Bars {
  uint64_t full[STG], empty[STG], acc_full;
  uint32_t tmem;
};
static_assert(sizeof(Bars) <= 512, "barrier block");

// maps: h [M][H] (box 64 x 128 x KBX), ws [D][H] (box 64 x NB / 2 x KBX; both kb_box maps). x / out [M][D] bf16. Grid: 2 x NBLK x nrb.
extern "C" __global__ void __launch_bounds__(256, 1)
transition_sx_squeeze2w(const __grid_constant__ CUtensorMap mh, const __grid_constant__ CUtensorMap mws, const ACT* __restrict__ xg,
                        ACT* __restrict__ out, int m TSX_STAMP_ARG) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  Bars& B = *reinterpret_cast<Bars*>(sm + O_BAR);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int pair = blockIdx.x >> 1, rank = (int)cluster_rank();
  const bool leader = rank == 0;
  const int rb = pair / NBLK, nb = pair % NBLK, arow = (rb * 2 + rank) * ROWS, c0 = nb * NB;
  if (tid == 0) TSX_ST(0);

  if (tid == 0) {
    for (int s = 0; s < STG; ++s) { mbar_init(&B.full[s], 1); mbar_init(&B.empty[s], 1); }
    mbar_init(&B.acc_full, 1);
    fence_barrier_init();
    prefetch_map(&mh); prefetch_map(&mws);
  }
  if (warp == 2) { tmem_alloc2(smem_u32(&B.tmem), TCOLS); tmem_relinquish2(); }
  tc_fence_before();
  __syncthreads();
  cluster_sync_relaxed();
  tc_fence_after();
  const uint32_t tmem = B.tmem;

  if (warp == 0) {
    if (lane < STG) {
      auto load_w = [&](int st) {                              // the stage's Ws K-blocks as ONE 3-D box (the per-SM box rate paces intake)
        if (KBX == 1) tma_load_2d_2sm(su + O_W + lane * WST, &mws, &B.full[lane], st * 64, c0 + rank * NBH);
        else tma_load_3d_2sm(su + O_W + lane * WST, &mws, &B.full[lane], 0, c0 + rank * NBH, st * KBX);
      };
      auto load_a = [&](int st) {
        if (KBX == 1) tma_load_2d_2sm(su + O_A + lane * AST, &mh, &B.full[lane], st * 64, arow);
        else tma_load_3d_2sm(su + O_A + lane * AST, &mh, &B.full[lane], 0, arow, st * KBX);
      };
      if (leader) mbar_expect_tx(&B.full[lane], 2 * (AST + WST));
      load_w(lane);                                            // weights: before the wait
      pdl_wait();
      if (lane == 0) TSX_ST(1);
      load_a(lane);
      for (int st = lane + STG; st < NS; st += STG) {
        mbar_wait(&B.empty[lane], ((st / STG) - 1) & 1);
        if (leader) mbar_expect_tx(&B.full[lane], 2 * (AST + WST));
        load_w(st);
        load_a(st);
      }
    }
    pdl_wait();
  } else if (warp == 1) {
    pdl_wait();
    if (leader) {
#pragma unroll 1
      for (int st = 0; st < NS; ++st) {
        const int s = st % STG;
        mbar_wait(&B.full[s], (st / STG) & 1);
        if (lane == 0 && st == 0) TSX_ST(2);
        tc_fence_after();
        if (elect_one()) {
#pragma unroll
          for (int i = 0; i < KBX; ++i) {
            const uint64_t da = desc_k128(su + O_A + s * AST + i * AKB), db = desc_k128(su + O_W + s * WST + i * WKB);
#pragma unroll
            for (int ks = 0; ks < 4; ++ks)
              umma_ss2(tmem, da + (uint64_t)(ks * 2), db + (uint64_t)(ks * 2), IDESC, (st > 0 || i > 0 || ks > 0) ? 1u : 0u);
          }
          if (NS > STG) tc_commit2_mc(&B.empty[s], 3);
          if (st == NS - 1) tc_commit2_mc(&B.acc_full, 3);
        }
        __syncwarp();
      }
      if (lane == 0) TSX_ST(3);
    }
  } else {
    pdl_wait();
  }
  pdl_launch();

  if (warp >= 4) {
    // ------------------------------------------------------------------------------------------ epilogue: one row per thread (the CTA's
    // 128 rows over warps 4-7), NB columns in passes of 32; + x, bf16
    const int r = (warp & 3) * 32 + lane;
    const size_t grow = (size_t)(arow + r);
    const bool whole = arow + ROWS <= m, live = arow + r < m;
    const uint32_t trow = tmem + ((uint32_t)((warp & 3) * 32) << 16);
    constexpr int NP = NB / 32;
    uint4 xv[NP][4];
#pragma unroll
    for (int p = 0; p < NP; ++p)
#pragma unroll
      for (int i = 0; i < 4; ++i) xv[p][i] = live ? ldg128(xg + grow * D_ + c0 + p * 32 + 8 * i) : make_uint4(0, 0, 0, 0);
    mbar_wait(&B.acc_full, 0);
    if (warp == 4 && lane == 0) TSX_ST(4);
    tc_fence_after();
#pragma unroll
    for (int p = 0; p < NP; ++p) {
      uint32_t v[32];
      tmem_ld32(trow + p * 32, v);
      tmem_wait_ld();
      uint32_t o[16];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const uint32_t xw[4] = {xv[p][i].x, xv[p][i].y, xv[p][i].z, xv[p][i].w};
#pragma unroll
        for (int k = 0; k < 4; ++k)
          o[4 * i + k] = pack_bf16(bf16lo(xw[k]) + __uint_as_float(v[8 * i + 2 * k]), bf16hi(xw[k]) + __uint_as_float(v[8 * i + 2 * k + 1]));
      }
      const int pc = c0 + p * 32;
      if (whole) {
        store_t(out + (size_t)(arow + (r & ~3)) * D_ + pc + (lane & 3) * 8, D_, o, lane);
      } else if (live) {
#pragma unroll
        for (int c = 0; c < 4; ++c) stg128(out + grow * D_ + pc + 8 * c, make_uint4(o[4 * c], o[4 * c + 1], o[4 * c + 2], o[4 * c + 3]));
      }
    }
    if (warp == 4 && lane == 0) TSX_ST(6);
    tc_fence_before();
  }
  __syncthreads();
  cluster_sync_relaxed();
  if (tid == 0) TSX_ST(7);
  if (warp == 2) { tc_fence_after(); tmem_dealloc2(tmem, TCOLS); }
}

extern "C" __device__ int transition_smem_bytes = SMEM_BYTES;
