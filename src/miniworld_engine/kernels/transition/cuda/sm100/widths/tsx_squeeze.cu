// tsx_squeeze.cu -- the Transition forward at SMALL M, second of two kernels: out = bf16(h Ws^T + x) (-DDIM=<D> -DHID=<H>
// -DSK=<K-split>), sm_100a, bf16, a programmatic dependent launch after tsx_expand.cu.
// One cluster of SK CTAs per (128-row tile, 64-column output block): tiles x D / 64 x SK CTAs (144 at [1, 768, 384] with SK = 4).
// CTA k of a cluster runs the hidden slice [k H / SK, (k + 1) H / SK) through a ring of (h, Ws) K-blocks into a [128 x 64] fp32
// accumulator in tensor memory. The partials are summed through L2: CTA o owns output columns [o 64 / SK, (o + 1) 64 / SK) of the
// block; every CTA stores the other owners' columns into their shared memory (distributed shared memory, st.async completing on the
// owner's barrier: no cluster barrier), and each owner adds the SK partials in rank order (deterministic), adds the residual (loaded
// while the products run) and stores bf16 -- one rounding.
// Ws K-blocks do not depend on the expand, so the first ring stages are loaded before griddepcontrol.wait.
// SPDX-License-Identifier: Apache-2.0
#include "sm100.cuh"
#include "tsx_store.cuh"
using namespace s100;

#ifndef DIM
#define DIM 384
#endif
#ifndef HID
#define HID (2 * DIM)
#endif
#ifndef SK
#define SK 1
#endif
#ifndef NBW
#define NBW 64                                                  // output columns per cluster: 32 or 64
#endif
constexpr int D_ = DIM, H_ = HID, ROWS = 128, NB = NBW, NBLK = D_ / NB, KS = H_ / SK, NKB = KS / 64, W_ = NB / SK;
static_assert(H_ % (64 * SK) == 0 && NB % SK == 0 && W_ % 8 == 0, "whole K-blocks per CTA, whole 16-byte owner column groups");
constexpr int STG = NKB < 6 ? NKB : 6;
constexpr int KBT = ROWS * 128, WBT = NB * 128;                 // h K-block [128][64] 16 KB, Ws K-block [NB rows][64]
constexpr int O_A = 0, O_B = STG * KBT, O_RED = O_B + STG * WBT;   // partials received from the other ranks: [SK][128][W] fp32
constexpr int O_BAR = O_RED + (SK > 1 ? SK * ROWS * W_ * 4 : 0), SMEM_BYTES = O_BAR + 256;
static_assert(SMEM_BYTES <= 232448, "shared memory budget");
constexpr uint32_t IDESC = idesc_bf16(128, NB);

struct Bars {
  uint64_t full[STG], empty[STG], acc_full, red_full;
  uint32_t tmem;
};

template <int N> DEVI void tld(uint32_t taddr, uint32_t (&r)[N]) {
  if constexpr (N == 32) tmem_ld32(taddr, r); else tmem_ld16(taddr, r);
}


// maps: h [M][H] (box 64 x 128), ws [D][H] (box 64 x NB). x / out [M][D] bf16.
extern "C" __global__ void __launch_bounds__(256, 1)
transition_sx_squeeze(const __grid_constant__ CUtensorMap mh, const __grid_constant__ CUtensorMap mws,
                      const __nv_bfloat16* __restrict__ xg, __nv_bfloat16* __restrict__ out) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  Bars& B = *reinterpret_cast<Bars*>(sm + O_BAR);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int cl = blockIdx.x / SK, rank = SK > 1 ? (int)cluster_rank() : 0, tile = cl / NBLK, nb = cl % NBLK;
  const int row0 = tile * ROWS, k0 = rank * KS;

  if (tid == 0) {
    for (int s = 0; s < STG; ++s) { mbar_init(&B.full[s], 1); mbar_init(&B.empty[s], 1); }
    mbar_init(&B.acc_full, 1);
    if (SK > 1) { mbar_init(&B.red_full, 1); mbar_expect_tx(&B.red_full, (SK - 1) * ROWS * W_ * 4); }   // the other ranks' partials
    fence_barrier_init();
    fence_proxy_async();
    prefetch_map(&mh); prefetch_map(&mws);
  }
  if (warp == 2) { tmem_alloc(smem_u32(&B.tmem), NB); tmem_relinquish(); }
  tc_fence_before();
  __syncthreads();
  if (SK > 1) cluster_sync_relaxed();                          // every CTA of the cluster runs before any distributed-shared store
  tc_fence_after();
  const uint32_t tmem = B.tmem;

  if (warp == 0) {
    if (lane == 0) {
      for (int kb = 0; kb < STG; ++kb) {                        // the weights first: they do not depend on the expand
        mbar_expect_tx(&B.full[kb], KBT + WBT);
        tma_load_2d(su + O_B + kb * WBT, &mws, &B.full[kb], k0 + kb * 64, nb * NB);
      }
      pdl_wait();
      for (int kb = 0; kb < STG; ++kb) tma_load_2d(su + O_A + kb * KBT, &mh, &B.full[kb], k0 + kb * 64, row0);
      for (int kb = STG; kb < NKB; ++kb) {
        const int s = kb % STG;
        mbar_wait(&B.empty[s], ((kb / STG) - 1) & 1);
        mbar_expect_tx(&B.full[s], KBT + WBT);
        tma_load_2d(su + O_B + s * WBT, &mws, &B.full[s], k0 + kb * 64, nb * NB);
        tma_load_2d(su + O_A + s * KBT, &mh, &B.full[s], k0 + kb * 64, row0);
      }
    }
    pdl_wait();
  } else if (warp == 1) {
    pdl_wait();
    for (int kb = 0; kb < NKB; ++kb) {
      const int s = kb % STG;
      mbar_wait(&B.full[s], (kb / STG) & 1);
      tc_fence_after();
      const uint64_t da = desc_k128(su + O_A + s * KBT), db = desc_k128(su + O_B + s * WBT);
      if (elect_one()) {
#pragma unroll
        for (int ks = 0; ks < 4; ++ks)
          umma_ss(tmem, da + (uint64_t)(ks * 2), db + (uint64_t)(ks * 2), IDESC, (kb > 0 || ks > 0) ? 1u : 0u);
        tc_commit(&B.empty[s]);
        if (kb == NKB - 1) tc_commit(&B.acc_full);
      }
      __syncwarp();
    }
  } else {
    pdl_wait();
  }
  pdl_launch();

  const int r = (warp & 3) * 32 + lane;
  const size_t grow = (size_t)(row0 + r);
  const uint32_t trow = tmem + ((uint32_t)((warp & 3) * 32) << 16);
  const int c0 = nb * NB + (SK > 1 ? rank * W_ : 0);             // this CTA's output columns: [c0, c0 + W) (SK > 1) or [c0, c0 + NB)
  constexpr int OW = SK > 1 ? W_ : NB;
  uint4 xv[OW / 8];                                            // the residual, loaded while the products run
  if (warp >= 4) {
#pragma unroll
    for (int i = 0; i < OW / 8; ++i) xv[i] = ldg128(xg + grow * D_ + c0 + 8 * i);
  }
#if SK == 1
  if (warp >= 4) {
    mbar_wait(&B.acc_full, 0);
    tc_fence_after();
#pragma unroll
    for (int cc = 0; cc < NB; cc += 32) {
      uint32_t v[32];
      tmem_ld32(trow + cc, v);
      tmem_wait_ld();
      uint32_t o[16];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const uint4 x4 = xv[cc / 8 + i];
        const uint32_t xw[4] = {x4.x, x4.y, x4.z, x4.w};
#pragma unroll
        for (int k = 0; k < 4; ++k)
          o[4 * i + k] = pack_bf16(bf16lo(xw[k]) + __uint_as_float(v[8 * i + 2 * k]), bf16hi(xw[k]) + __uint_as_float(v[8 * i + 2 * k + 1]));
      }
      store_t(out + (size_t)row0 * D_ + c0 + cc + store_t_off<16>(r, lane, D_, 2), D_, o, lane);
    }
    tc_fence_before();
  }
  __syncthreads();
#else
  float acc[W_];
  if (warp >= 4) {
    // ------------------------------------------------------------------------------------------ the other owners' columns go
    // straight into their shared memory (distributed shared memory), [source rank][row][W] fp32
    mbar_wait(&B.acc_full, 0);
    tc_fence_after();
#pragma unroll
    for (int oo = 1; oo <= SK; ++oo) {
      const int o = (rank + oo) % SK;                          // own block last; the others staggered over the cluster
      uint32_t v[W_];
      tld<W_>(trow + o * W_, v);
      tmem_wait_ld();
      if (o == rank) {
#pragma unroll
        for (int i = 0; i < W_; ++i) acc[i] = __uint_as_float(v[i]);
      } else {
        const uint32_t dst = su + O_RED + (uint32_t)(rank * ROWS * W_ * 4);
#pragma unroll
        for (int i = 0; i < W_ / 4; ++i)
          st_async_v4(dst + swz16((uint32_t)(r * (W_ / 4) + i)), smem_u32(&B.red_full), (uint32_t)o, v[4 * i], v[4 * i + 1], v[4 * i + 2],
                      v[4 * i + 3]);
      }
    }
    tc_fence_before();
  }
  if (warp >= 4) {
    mbar_wait(&B.red_full, 0);                                 // every other rank's partial of these columns is here
    float sum[W_];
#pragma unroll
    for (int i = 0; i < W_; ++i) sum[i] = 0.f;
#pragma unroll
    for (int o = 0; o < SK; ++o) {                             // fixed order over the hidden slices: deterministic
      if (o == rank) {
#pragma unroll
        for (int i = 0; i < W_; ++i) sum[i] += acc[i];
      } else {
        const uint32_t src = su + O_RED + (uint32_t)(o * ROWS * W_ * 4);
#pragma unroll
        for (int i = 0; i < W_ / 4; ++i) {
          const uint4 t = lds128(src + swz16((uint32_t)(r * (W_ / 4) + i)));
          sum[4 * i] += __uint_as_float(t.x); sum[4 * i + 1] += __uint_as_float(t.y);
          sum[4 * i + 2] += __uint_as_float(t.z); sum[4 * i + 3] += __uint_as_float(t.w);
        }
      }
    }
    uint32_t o[W_ / 2];
#pragma unroll
    for (int i = 0; i < W_ / 8; ++i) {
      const uint32_t xw[4] = {xv[i].x, xv[i].y, xv[i].z, xv[i].w};
#pragma unroll
      for (int k = 0; k < 4; ++k) o[4 * i + k] = pack_bf16(bf16lo(xw[k]) + sum[8 * i + 2 * k], bf16hi(xw[k]) + sum[8 * i + 2 * k + 1]);
    }
    store_t(out + (size_t)row0 * D_ + c0 + store_t_off<W_ / 2>(r, lane, D_, 2), D_, o, lane);
  }
#endif
  __syncthreads();
  if (warp == 2) { tc_fence_after(); tmem_dealloc(tmem, NB); }
}

extern "C" __device__ int transition_smem_bytes = SMEM_BYTES;
