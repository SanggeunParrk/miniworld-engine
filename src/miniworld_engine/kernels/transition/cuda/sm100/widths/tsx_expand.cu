// tsx_expand.cu -- the Transition forward at SMALL M (a few 128-row tiles: the single track [B, L, D] of a pairformer), second of
// three programmatic-dependent kernels (tsx_ln.cu -> this -> tsx_squeeze.cu): expand + SwiGLU (-DDIM=<D> -DHID=<H> -DCH=<hidden
// units per CTA, 32 or 64>), sm_100a, bf16.
// One CTA per (128-row tile, CH-unit hidden chunk j): tiles x H / CH CTAs, sized so that a [1, 768, 384] call fills the 148 SMs (6 x 24
// at n = 2 with CH = 32, 6 x 24 at n = 4 with CH = 64). Each CTA TMA-loads the tile's xn (D x 128 bf16, written by tsx_ln.cu) and its
// chunk of Wa and Wb (CH rows each), K-block by K-block with a barrier each so the products start as the data lands, runs
// [a | b] = xn [Wa_j; Wb_j]^T (M 128, N 2 CH, K D) into tensor memory, and writes h = bf16(silu(a) b) (and, with save, a and b for the
// backward). No cross-CTA reduction: h goes through L2 (M x H bf16) to the squeeze.
// The weight K-blocks are loaded before griddepcontrol.wait when they are parameters or an earlier launch's cast (wpre = 1); when the
// LayerNorm launch this one follows writes them (the training forward's fused cast, wpre = 0), after it.
// SPDX-License-Identifier: Apache-2.0
#include "sm100.cuh"
using namespace s100;

#ifndef DIM
#define DIM 384
#endif
#ifndef HID
#define HID (2 * DIM)
#endif
#ifndef CH
#define CH 64
#endif
constexpr int D_ = DIM, H_ = HID, ROWS = 128, NKB = D_ / 64, NCHK = H_ / CH, HC = CH / 2;
static_assert((CH == 32 || CH == 64) && H_ % CH == 0 && D_ % 64 == 0, "32- or 64-unit chunks, whole K-blocks");
constexpr int KBT = ROWS * 128;                                // one [128 rows][64] bf16 K-block, 16 KB
constexpr int WKB = 2 * CH * 128;                              // one K-block of [Wa_j; Wb_j]: 2 CH rows x 128 B
constexpr int O_X = 0, O_W = NKB * KBT, O_BAR = O_W + NKB * WKB, SMEM_BYTES = O_BAR + 256;
static_assert(SMEM_BYTES <= 232448, "shared memory budget");
constexpr uint32_t TCOLS = 2 * CH, IDESC = idesc_bf16(128, 2 * CH);

struct Bars {
  uint64_t full[NKB], mma_done;
  uint32_t tmem;
};

template <int N> DEVI void tld(uint32_t taddr, uint32_t (&r)[N]) {
  if constexpr (N == 32) tmem_ld32(taddr, r); else tmem_ld16(taddr, r);
}

// maps: xn [M][D] (box 64 x 128); wa, wb [H][D] (box 64 x CH). h / a / b [M][H] bf16.
extern "C" __global__ void __launch_bounds__(384, 1)
transition_sx_expand(const __grid_constant__ CUtensorMap mxn, const __grid_constant__ CUtensorMap mwa,
                     const __grid_constant__ CUtensorMap mwb, __nv_bfloat16* __restrict__ hs, __nv_bfloat16* __restrict__ as_,
                     __nv_bfloat16* __restrict__ bs, int save, int wpre) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  Bars& B = *reinterpret_cast<Bars*>(sm + O_BAR);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int tile = blockIdx.x / NCHK, j = blockIdx.x % NCHK, row0 = tile * ROWS, h0 = j * CH;

  if (tid == 0) {
    for (int kb = 0; kb < NKB; ++kb) mbar_init(&B.full[kb], 1);
    mbar_init(&B.mma_done, 1);
    fence_barrier_init();
    fence_proxy_async();
    prefetch_map(&mxn); prefetch_map(&mwa); prefetch_map(&mwb);
    if (wpre)
      for (int kb = 0; kb < NKB; ++kb) {                        // weights first: they do not depend on the LayerNorm kernel
        mbar_expect_tx(&B.full[kb], KBT + WKB);
        tma_load_2d(su + O_W + kb * WKB, &mwa, &B.full[kb], kb * 64, h0);
        tma_load_2d(su + O_W + kb * WKB + CH * 128, &mwb, &B.full[kb], kb * 64, h0);
      }
  }
  if (warp == 2) { tmem_alloc(smem_u32(&B.tmem), TCOLS); tmem_relinquish(); }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = B.tmem;
  pdl_wait();                                                  // xn is complete and visible
  pdl_launch();                                                // the squeeze may start its prologue

  if (warp == 0) {
    if (lane == 0)
      for (int kb = 0; kb < NKB; ++kb) {
        if (!wpre) {                                           // the LayerNorm launch wrote the weights: complete only now
          mbar_expect_tx(&B.full[kb], KBT + WKB);
          tma_load_2d(su + O_W + kb * WKB, &mwa, &B.full[kb], kb * 64, h0);
          tma_load_2d(su + O_W + kb * WKB + CH * 128, &mwb, &B.full[kb], kb * 64, h0);
        }
        tma_load_2d(su + O_X + kb * KBT, &mxn, &B.full[kb], kb * 64, row0);
      }
  } else if (warp == 1) {
    // ------------------------------------------------------------------------------------------ MMA issue (converged warp)
#pragma unroll 1
    for (int kb = 0; kb < NKB; ++kb) {
      mbar_wait(&B.full[kb], 0);
      tc_fence_after();
      const uint64_t da = desc_k128(su + O_X + kb * KBT), db = desc_k128(su + O_W + kb * WKB);
      if (elect_one()) {
#pragma unroll
        for (int ks = 0; ks < 4; ++ks)
          umma_ss(tmem, da + (uint64_t)(ks * 2), db + (uint64_t)(ks * 2), IDESC, (kb > 0 || ks > 0) ? 1u : 0u);
        if (kb == NKB - 1) tc_commit(&B.mma_done);
      }
      __syncwarp();
    }
  } else if (warp >= 4) {
    // ------------------------------------------------------------------------------------------ SwiGLU (one row per thread, warps
    // 4-7 the chunk's first CH / 2 units, 8-11 the second)
    const int r = (warp & 3) * 32 + lane, g = (warp - 4) >> 2;
    const uint32_t trow = tmem + ((uint32_t)((warp & 3) * 32) << 16);
    const size_t off = (size_t)(row0 + r) * H_ + h0 + g * HC;
    mbar_wait(&B.mma_done, 0);
    tc_fence_after();
    uint32_t av[HC], bv[HC], hp[HC / 2], ap[HC / 2], bp[HC / 2];
    tld<HC>(trow + g * HC, av);
    tld<HC>(trow + CH + g * HC, bv);
    tmem_wait_ld();
    tc_fence_before();
#pragma unroll
    for (int k = 0; k < HC / 2; ++k) {
      const float a0 = __uint_as_float(av[2 * k]), a1 = __uint_as_float(av[2 * k + 1]);
      const float b0 = __uint_as_float(bv[2 * k]), b1 = __uint_as_float(bv[2 * k + 1]);
      hp[k] = pack_bf16(a0 * sigmoid_kit(a0) * b0, a1 * sigmoid_kit(a1) * b1);
      ap[k] = pack_bf16(a0, a1);
      bp[k] = pack_bf16(b0, b1);
    }
#pragma unroll
    for (int q = 0; q < HC / 8; ++q)
      stg128(hs + off + q * 8, make_uint4(hp[4 * q], hp[4 * q + 1], hp[4 * q + 2], hp[4 * q + 3]));
    if (save) {
#pragma unroll
      for (int q = 0; q < HC / 8; ++q) {
        stg128(as_ + off + q * 8, make_uint4(ap[4 * q], ap[4 * q + 1], ap[4 * q + 2], ap[4 * q + 3]));
        stg128(bs + off + q * 8, make_uint4(bp[4 * q], bp[4 * q + 1], bp[4 * q + 2], bp[4 * q + 3]));
      }
    }
  }
  __syncthreads();
  if (warp == 2) { tc_fence_after(); tmem_dealloc(tmem, TCOLS); }
}

extern "C" __device__ int transition_smem_bytes = SMEM_BYTES;
