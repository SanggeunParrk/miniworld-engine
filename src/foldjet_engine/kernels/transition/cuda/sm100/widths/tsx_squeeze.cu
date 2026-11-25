// tsx_squeeze.cu -- the Transition forward at SMALL M, last kernel: out = h Ws^T + x (-DDIM=<D> -DHID=<H> -DF32=<0 bf16 | 1 fp32 on TF32>
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
#ifndef F32
#define F32 0
#endif
#ifndef H16
#define H16 0                                                   // round 3 part 4: h / Ws16 on the fp16 tensor cores, fp32 x / out, y = acc sc[3] + x
#endif
static_assert(!(H16 && F32), "H16 is its own operand type");
#ifndef CHM
#define CHM 1                                                   // part 17: partials received chunk-major (512-B runs per warp store); 0: row-major swizzled
#endif
#if CHM
#define RXO(i) chm16((uint32_t)(i), (uint32_t)r)
#else
#define RXO(i) swz16((uint32_t)(r * (W_ / 4) + (i)))
#endif
constexpr int IOSZ = (F32 || H16) ? 4 : 2;                     // x / out element bytes
#if H16
#define SC(v) ((v) * s3)
#else
#define SC(v) (v)
#endif
constexpr int ESZ = F32 ? 4 : 2, BKE = 128 / ESZ;              // element bytes, elements of K per 128-byte K-block row
constexpr int D_ = DIM, H_ = HID, ROWS = 128, NB = NBW, NBLK = D_ / NB, KS = H_ / SK, NKB = KS / BKE, W_ = NB / SK;
static_assert(H_ % (BKE * SK) == 0 && NB % SK == 0 && (W_ * IOSZ) % 32 == 0, "whole K-blocks per CTA, owner column groups of 32 bytes or more");
constexpr int STG = NKB < 6 ? NKB : 6;
constexpr int KBT = ROWS * 128, WBT = NB * 128;                 // h K-block [128][64] 16 KB, Ws K-block [NB rows][64]
constexpr int O_A = 0, O_B = STG * KBT, O_RED = O_B + STG * WBT;   // partials received from the other ranks: [SK][128][W] fp32
constexpr int O_BAR = O_RED + (SK > 1 ? SK * ROWS * W_ * 4 : 0), SMEM_BYTES = O_BAR + 256;
static_assert(SMEM_BYTES <= 232448, "shared memory budget");
#if F32
typedef float ACT;
// kind::tf32 instruction descriptor: TF32 A / B (format 2), fp32 D, K-major both, N, M
constexpr uint32_t IDESC = (1u << 4) | (2u << 7) | (2u << 10) | ((uint32_t)(NB >> 3) << 17) | ((uint32_t)(128 >> 4) << 24);
DEVI void mma(uint32_t d, uint64_t a, uint64_t b, uint32_t acc) {
  asm volatile("{ .reg .pred p; setp.ne.b32 p, %4, 0; tcgen05.mma.cta_group::1.kind::tf32 [%0], %1, %2, %3, p; }"
               :: "r"(d), "l"(a), "l"(b), "r"(IDESC), "r"(acc) : "memory");
}
#elif H16
typedef float ACT;                                             // x / out; the operands are fp16 (kind::f16, format 0)
constexpr uint32_t IDESC = (1u << 4) | ((uint32_t)(NB >> 3) << 17) | ((uint32_t)(128 >> 4) << 24);
DEVI void mma(uint32_t d, uint64_t a, uint64_t b, uint32_t acc) { umma_ss(d, a, b, IDESC, acc); }
#else
typedef __nv_bfloat16 ACT;
constexpr uint32_t IDESC = idesc_bf16(128, NB);
DEVI void mma(uint32_t d, uint64_t a, uint64_t b, uint32_t acc) { umma_ss(d, a, b, IDESC, acc); }
#endif

struct Bars {
  uint64_t full[STG], empty[STG], acc_full, red_full;
  uint32_t tmem;
};

template <int N> DEVI void tld(uint32_t taddr, uint32_t (&r)[N]) {
  if constexpr (N == 32) tmem_ld32(taddr, r);
  else if constexpr (N == 16) tmem_ld16(taddr, r);
  else
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]) : "r"(taddr) : "memory");
}


// maps: h [M][H] (box 64 x 128), ws [D][H] (box 64 x NB). x / out [M][D] bf16.
extern "C" __global__ void __launch_bounds__(256, 1)
transition_sx_squeeze(const __grid_constant__ CUtensorMap mh, const __grid_constant__ CUtensorMap mws,
                      const ACT* __restrict__ xg, ACT* __restrict__ out
#if H16
                      , const float* __restrict__ sc
#endif
                      TSX_STAMP_ARG
                      ) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  Bars& B = *reinterpret_cast<Bars*>(sm + O_BAR);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int cl = blockIdx.x / SK, rank = SK > 1 ? (int)cluster_rank() : 0, tile = cl / NBLK, nb = cl % NBLK;
  const int row0 = tile * ROWS, k0 = rank * KS;
  if (tid == 0) TSX_ST(0);

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
        tma_load_2d(su + O_B + kb * WBT, &mws, &B.full[kb], k0 + kb * BKE, nb * NB);
      }
      pdl_wait();
      TSX_ST(1);
      for (int kb = 0; kb < STG; ++kb) tma_load_2d(su + O_A + kb * KBT, &mh, &B.full[kb], k0 + kb * BKE, row0);
      for (int kb = STG; kb < NKB; ++kb) {
        const int s = kb % STG;
        mbar_wait(&B.empty[s], ((kb / STG) - 1) & 1);
        mbar_expect_tx(&B.full[s], KBT + WBT);
        tma_load_2d(su + O_B + s * WBT, &mws, &B.full[s], k0 + kb * BKE, nb * NB);
        tma_load_2d(su + O_A + s * KBT, &mh, &B.full[s], k0 + kb * BKE, row0);
      }
    }
    pdl_wait();
  } else if (warp == 1) {
    pdl_wait();
    for (int kb = 0; kb < NKB; ++kb) {
      const int s = kb % STG;
      mbar_wait(&B.full[s], (kb / STG) & 1);
      if (lane == 0 && kb == 0) TSX_ST(2);
      if (lane == 0 && kb == NKB - 1) TSX_ST(3);
      tc_fence_after();
      const uint64_t da = desc_k128(su + O_A + s * KBT), db = desc_k128(su + O_B + s * WBT);
      if (elect_one()) {
#pragma unroll
        for (int ks = 0; ks < 4; ++ks)
          mma(tmem, da + (uint64_t)(ks * 2), db + (uint64_t)(ks * 2), (kb > 0 || ks > 0) ? 1u : 0u);
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
#if H16
  const float s3 = __ldg(sc + 3);
#endif
  uint4 xv[OW * IOSZ / 16];                                        // the residual, loaded while the products run
  if (warp >= 4) {
#pragma unroll
    for (int i = 0; i < OW * IOSZ / 16; ++i) xv[i] = ldg128(xg + grow * D_ + c0 + (16 / IOSZ) * i);
  }
#if SK == 1
  if (warp >= 4) {
    mbar_wait(&B.acc_full, 0);
    if (warp == 4 && lane == 0) TSX_ST(4);
    tc_fence_after();
#pragma unroll
    for (int cc = 0; cc < NB; cc += 32) {
      uint32_t v[32];
      tmem_ld32(trow + cc, v);
      tmem_wait_ld();
#if F32 || H16
      uint32_t o[32];
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const uint4 x4 = xv[cc / 4 + i];
        o[4 * i] = __float_as_uint(__uint_as_float(x4.x) + SC(__uint_as_float(v[4 * i])));
        o[4 * i + 1] = __float_as_uint(__uint_as_float(x4.y) + SC(__uint_as_float(v[4 * i + 1])));
        o[4 * i + 2] = __float_as_uint(__uint_as_float(x4.z) + SC(__uint_as_float(v[4 * i + 2])));
        o[4 * i + 3] = __float_as_uint(__uint_as_float(x4.w) + SC(__uint_as_float(v[4 * i + 3])));
      }
      store_t(out + (size_t)row0 * D_ + c0 + cc + store_t_off<32>(r, lane, D_, 4), D_, o, lane);
#else
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
#endif
    }
    if (warp == 4 && lane == 0) TSX_ST(6);
    tc_fence_before();
  }
  __syncthreads();
  if (tid == 0) TSX_ST(7);
#else
  float acc[W_];
  if (warp >= 4) {
    // ------------------------------------------------------------------------------------------ the other owners' columns go
    // straight into their shared memory (distributed shared memory), [source rank][row][W] fp32
    mbar_wait(&B.acc_full, 0);
    if (warp == 4 && lane == 0) TSX_ST(4);
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
          st_async_v4(dst + RXO(i), smem_u32(&B.red_full), (uint32_t)o, v[4 * i], v[4 * i + 1], v[4 * i + 2],
                      v[4 * i + 3]);
      }
    }
    tc_fence_before();
  }
  if (warp >= 4) {
    mbar_wait(&B.red_full, 0);                                 // every other rank's partial of these columns is here
    if (warp == 4 && lane == 0) TSX_ST(5);
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
          const uint4 t = lds128(src + RXO(i));
          sum[4 * i] += __uint_as_float(t.x); sum[4 * i + 1] += __uint_as_float(t.y);
          sum[4 * i + 2] += __uint_as_float(t.z); sum[4 * i + 3] += __uint_as_float(t.w);
        }
      }
    }
#if F32 || H16
    uint32_t o[W_];
#pragma unroll
    for (int i = 0; i < W_ / 4; ++i) {
      o[4 * i] = __float_as_uint(__uint_as_float(xv[i].x) + SC(sum[4 * i])); o[4 * i + 1] = __float_as_uint(__uint_as_float(xv[i].y) + SC(sum[4 * i + 1]));
      o[4 * i + 2] = __float_as_uint(__uint_as_float(xv[i].z) + SC(sum[4 * i + 2])); o[4 * i + 3] = __float_as_uint(__uint_as_float(xv[i].w) + SC(sum[4 * i + 3]));
    }
    store_t(out + (size_t)row0 * D_ + c0 + store_t_off<W_>(r, lane, D_, 4), D_, o, lane);
#else
    uint32_t o[W_ / 2];
#pragma unroll
    for (int i = 0; i < W_ / 8; ++i) {
      const uint32_t xw[4] = {xv[i].x, xv[i].y, xv[i].z, xv[i].w};
#pragma unroll
      for (int k = 0; k < 4; ++k) o[4 * i + k] = pack_bf16(bf16lo(xw[k]) + sum[8 * i + 2 * k], bf16hi(xw[k]) + sum[8 * i + 2 * k + 1]);
    }
    store_t(out + (size_t)row0 * D_ + c0 + store_t_off<W_ / 2>(r, lane, D_, 2), D_, o, lane);
#endif
    if (warp == 4 && lane == 0) TSX_ST(6);
  }
#endif
  __syncthreads();
  if (tid == 0) TSX_ST(7);
  if (warp == 2) { tc_fence_after(); tmem_dealloc(tmem, NB); }
}

extern "C" __device__ int transition_smem_bytes = SMEM_BYTES;
