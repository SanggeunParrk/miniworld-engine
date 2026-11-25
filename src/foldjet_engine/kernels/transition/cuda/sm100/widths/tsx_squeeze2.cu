// tsx_squeeze2.cu -- the small-M Transition squeeze out = h Ws^T + x (tsx_squeeze.cu's place in the chain) on a CTA PAIR
// (cta_group::2), sm_100a, bf16, inference (-DDIM=<D> -DHID=<H> -DNBW=<output columns per pair: 16 / 32 / 64>).
// One 2-CTA cluster per (128-row tile, NBW-column output block): tiles x D / NBW x 2 CTAs. The leader issues M = 128, N = NBW products
// over the whole hidden width: CTA r holds rows [64 r, 64 r + 64) of the tile's h and Ws rows [c0 + r NBW / 2, c0 + (r + 1) NBW / 2), and
// receives its 64 rows of the whole [128 x NBW] block, folded over the lanes (lanes 0-63 the leader's columns, 64-127 the peer's: the
// layout tsx_expand2.cu measured). No hidden split, so no partial exchange (tsx_squeeze.cu -DSK=4 spends ~1.5 us of a [1, 128, 768] call
// on it); each SM takes half the tile's h. The Ws K-blocks of the first ring round are loaded before griddepcontrol.wait.
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
#define NBW 32
#endif
constexpr int D_ = DIM, H_ = HID, ROWS = 128, HR = 64, NB = NBW, NBH = NB / 2, NBLK = D_ / NB, NKB = H_ / 64;
static_assert((NB == 16 || NB == 32 || NB == 64 || NB == 128) && D_ % NB == 0 && H_ % 64 == 0, "16- / 32- / 64- / 128-column blocks, whole K-blocks");
constexpr int AKB = HR * 128, WKB = NBH * 128;                  // one K-block of this CTA's 64 h rows (8 KB) / of its NB / 2 Ws rows
#ifndef KBX
#define KBX 1                                                   // K-blocks per ring stage: the h rows of a stage in ONE 3-D TMA box
#endif
constexpr int NS = NKB / KBX, AST = KBX * AKB, WST = KBX * WKB;
static_assert(NKB % KBX == 0, "whole stages");
constexpr int CAP = (232448 - 1024) / (AST + WST);
constexpr int STG = (NS < CAP ? NS : CAP) < 32 ? (NS < CAP ? NS : CAP) : 32;
static_assert(STG >= 2, "ring stages");
constexpr int O_A = 0, O_W = STG * AST, O_BAR = O_W + STG * WST, SMEM_BYTES = O_BAR + 512;
DEVI void tma_load_3d_2sm(uint32_t dst, const CUtensorMap* m, uint64_t* bar, int c0, int c1, int c2) {
  asm volatile("cp.async.bulk.tensor.3d.cta_group::2.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5}], [%2];"
               :: "r"(dst), "l"(m), "r"(smem_u32(bar) & 0xFEFFFFFFu), "r"(c0), "r"(c1), "r"(c2) : "memory");
}
static_assert(SMEM_BYTES <= 232448, "shared memory budget");
constexpr uint32_t TCOLS = NBH > 32 ? NBH : 32;               // NB / 2 fp32 columns (at least 32: the allocation granule)
#ifndef H16
#define H16 0                                                   // part 17: fp32 inference on the fp16 tensor cores (tsx_squeeze.cu -DH16's
#endif                                                          // scheme: h / Ws16 fp16, fp32 x / out, y = acc sc[3] + x)
#if H16
constexpr uint32_t IDESC = (1u << 4) | ((uint32_t)(NB >> 3) << 17) | ((uint32_t)(128 >> 4) << 24);   // fp16 A / B (format 0), fp32 D
typedef float ACT;
#else
constexpr uint32_t IDESC = idesc_bf16(128, NB);
typedef __nv_bfloat16 ACT;
#endif
constexpr int IOSZ = sizeof(ACT);

template <int N> DEVI void tld(uint32_t taddr, uint32_t (&r)[N]) {
  if constexpr (N == 32) tmem_ld32(taddr, r);
  else if constexpr (N == 16) tmem_ld16(taddr, r);
  else
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]) : "r"(taddr) : "memory");
}

template <int NW> DEVI void st_row(ACT* grp, ACT* own, const uint32_t (&o)[NW], int lane) {
  if constexpr (NW >= 8) store_t(grp, D_, o, lane);
  else stg128(own, make_uint4(o[0], o[1], o[2], o[3]));
}

struct Bars {
  uint64_t full[STG], empty[STG], acc_full;
  uint32_t tmem;
};
static_assert(sizeof(Bars) <= 512, "barrier block");

// maps: h [M][H] (box 64 x 64), ws [D][H] (box 64 x NB / 2). x / out [M][D] bf16.
extern "C" __global__ void __launch_bounds__(256, 1)
transition_sx_squeeze2(const __grid_constant__ CUtensorMap mh, const __grid_constant__ CUtensorMap mws, const ACT* __restrict__ xg,
                       ACT* __restrict__ out
#if H16
                       , const float* __restrict__ sc
#endif
                       TSX_STAMP_ARG) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  Bars& B = *reinterpret_cast<Bars*>(sm + O_BAR);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int pair = blockIdx.x >> 1, rank = (int)cluster_rank();
  const bool leader = rank == 0;
  const int tile = pair / NBLK, nb = pair % NBLK, arow = tile * ROWS + rank * HR, c0 = nb * NB;
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
      auto load_w = [&](int st) {
#pragma unroll
        for (int i = 0; i < KBX; ++i)
          tma_load_2d_2sm(su + O_W + lane * WST + i * WKB, &mws, &B.full[lane], (st * KBX + i) * 64, c0 + rank * NBH);
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
        if (lane == 0 && st == NS - 1) TSX_ST(3);
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
    }
  } else {
    pdl_wait();
  }
  pdl_launch();

  if (warp >= 4) {
    // ------------------------------------------------------------------------------------------ epilogue: lanes 0-63 of tensor memory
    // (warps 4, 5) the leader's NB / 2 columns of this CTA's 64 rows, lanes 64-127 (warps 6, 7) the peer's; + x, bf16
    const int q = warp & 3, r = (q & 1) * 32 + lane;
    const int col = c0 + (q >> 1) * NBH;
    const size_t grow = (size_t)(arow + r);
    const uint32_t trow = tmem + ((uint32_t)(q * 32) << 16);
    constexpr int CW = NBH < 32 ? NBH : 32, NCW = NBH / CW;        // columns per epilogue pass (NB 128: two passes of 32)
    constexpr int NX = CW * IOSZ / 16;                            // 16-byte residual chunks of a pass's row segment
    uint4 xv[NCW][NX];
#pragma unroll
    for (int p = 0; p < NCW; ++p)
#pragma unroll
      for (int i = 0; i < NX; ++i) xv[p][i] = ldg128(xg + grow * D_ + col + p * CW + (16 / IOSZ) * i);
#if H16
    const float s3 = __ldg(sc + 3);
#endif
    mbar_wait(&B.acc_full, 0);
    if (warp == 4 && lane == 0) TSX_ST(4);
    tc_fence_after();
#pragma unroll
    for (int p = 0; p < NCW; ++p) {
      uint32_t v[CW];
      tld(trow + p * CW, v);
      tmem_wait_ld();
      constexpr int NW = CW * IOSZ / 4;
      uint32_t o[NW];
#pragma unroll
      for (int i = 0; i < NX; ++i) {
        const uint32_t xw[4] = {xv[p][i].x, xv[p][i].y, xv[p][i].z, xv[p][i].w};
#pragma unroll
        for (int k = 0; k < 4; ++k)
#if H16
          o[4 * i + k] = __float_as_uint(__uint_as_float(xw[k]) + __uint_as_float(v[4 * i + k]) * s3);
#else
          o[4 * i + k] = pack_bf16(bf16lo(xw[k]) + __uint_as_float(v[8 * i + 2 * k]), bf16hi(xw[k]) + __uint_as_float(v[8 * i + 2 * k + 1]));
#endif
      }
      const int pc = col + p * CW;
      st_row<NW>(out + (size_t)(arow + (r & ~(NW / 4 - 1))) * D_ + pc + (lane & (NW / 4 - 1)) * (16 / IOSZ), out + grow * D_ + pc, o, lane);
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
