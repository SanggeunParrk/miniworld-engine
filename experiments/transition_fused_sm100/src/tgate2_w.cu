// tgate2_w.cu — the wide-width backward gate (D = 256 / 384 / 512, H = 4D; -DDIM=<D>) with TWO 64-unit hidden chunks per A load:
// the tgate_w.cu contract (dh = bf16(dy Ws), [a|b] = xn [Wa; Wb]^T in fp32, kit sigmoid; h, dA, dB bf16), per (128-row tile, chunk pair p =
// hidden units 128 p .. 128 p + 127). tgate_w.cu re-streamed A (dy, xn) for every 64-unit chunk, and with the h / dA / dB stores the kernel
// was bound by the SM's TMA traffic (~52 B/clk measured with no stores at all); one A load now feeds 128 units.
//
// 2-CTA tcgen05.mma, M = 256 (each CTA its own 128-row tile), B split by N over the pair:
//   dh   N 128: the leader holds Ws^T rows of chunk 2p, the peer of chunk 2p + 1         -> TMEM columns 256 .. 383
//   a|b  N 256: the leader holds Wa rows 128 p .. +127, the peer Wb rows                  -> TMEM columns 0 .. 127 (a), 128 .. 255 (b)
// One accumulator buffer (384 columns). The epilogue: warpgroup g (warps 4-7: g = 0, 12-15: g = 1) takes chunk 2p + g in two passes of 32
// units and releases the accumulator right after its second pass's TMEM loads; each warpgroup stages its chunk's h, dA, dB (48 KB) and a
// store thread TMA-stores them. SPDX-License-Identifier: Apache-2.0
#include "sm100.cuh"
using namespace s100;

#ifndef DIM
#define DIM 256
#endif
#ifndef NST_
#define NST_ 2
#endif
constexpr int D_ = DIM, H_ = 4 * DIM, HS = 64, NPAIR = H_ / 128, ROWS = 128, NKB = D_ / 64;
constexpr int NST = NST_;
constexpr int KBT = ROWS * 128;                                // one K-block of a 128-row tile: 16 KB
constexpr int S_DY = 0, S_XN = KBT, S_WAB = 2 * KBT, S_WS = 3 * KBT, STAGE = 3 * KBT + 8192;   // 56 KB
constexpr int O_ST = 0, O_OUT = NST * STAGE;                   // staging: per warpgroup h | dA | dB, [128][64] each
constexpr int O_BAR = O_OUT + 2 * 3 * KBT;
constexpr int SMEM_BYTES = O_BAR + 512;
static_assert(SMEM_BYTES <= 232448, "shared memory budget");
static_assert(STAGE % 1024 == 0 && O_OUT % 1024 == 0, "1 KB alignment of swizzled tiles");
constexpr uint32_t IDESC_DH = idesc_bf16(256, 128), IDESC_AB = idesc_bf16(256, 256);
constexpr uint32_t T_AB = 0, T_DH = 256;

struct Bars {
  uint64_t full[NST];                                          // leader: both CTAs' stage transactions
  uint64_t empty[NST], acc_full;                               // both CTAs (leader's multicast commits)
  uint64_t acc_empty;                                          // leader: 8 epilogue warps per CTA
  uint64_t staged[2], stage_free[2];                           // local, per warpgroup
  uint32_t tmem;
};

DEVI void gate_pair16(uint32_t dh0, uint32_t dh1, uint32_t a0, uint32_t a1, uint32_t b0, uint32_t b1, uint32_t& hp, uint32_t& dap, uint32_t& dbp) {
  const uint32_t gp = pack_bf16(__uint_as_float(dh0), __uint_as_float(dh1));
  const f2 G = mk2(bf16lo(gp), bf16hi(gp)), A = mk2u(a0, a1), Bv = mk2u(b0, b1);
  const f2 S = mk2(sigmoid_kit(__uint_as_float(a0)), sigmoid_kit(__uint_as_float(a1)));
  const f2 L = mul2(A, S);
  const f2 H = mul2(L, Bv), DB = mul2(G, L);
  const f2 U = fma2(L, fma2(S, mk2(-1.f, -1.f), mk2(1.f, 1.f)), S);
  const f2 DA = mul2(mul2(G, Bv), U);
  hp = pack_bf16(lo2(H), hi2(H)); dap = pack_bf16(lo2(DA), hi2(DA)); dbp = pack_bf16(lo2(DB), hi2(DB));
}

// maps: dy, xn [M][D] (box 64 x 64); wst = Ws^T [H][D], wa, wb [H][D] (box 64 x 64 / 64 x 128 for wa, wb); h [M][H], dab [M][2H] (box 64 x 64)
extern "C" __global__ void __launch_bounds__(512, 1)
transition_gate2_w_sm100(const __grid_constant__ CUtensorMap mdy, const __grid_constant__ CUtensorMap mxn,
                         const __grid_constant__ CUtensorMap mwst, const __grid_constant__ CUtensorMap mwa,
                         const __grid_constant__ CUtensorMap mwb, const __grid_constant__ CUtensorMap mh,
                         const __grid_constant__ CUtensorMap mdab, int tiles, int store_h) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  Bars& B = *reinterpret_cast<Bars*>(sm + O_BAR);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int cta = blockIdx.x, G = gridDim.x;
  const int crank = (int)cluster_rank();
  const bool leader = crank == 0;
  auto count = [&](int k) { return (tiles > k) ? (tiles - k + G - 1) / G : 0; };
  const int n_valid = count(cta), n_local = count(cta & ~1);
  const int nitem = n_local * NPAIR;
  auto tile_of = [&](int i) { return i < n_valid ? cta + i * G : (n_valid > 0 ? cta + (n_valid - 1) * G : 0); };

  if (tid == 0) {
    for (int s = 0; s < NST; ++s) { mbar_init(&B.full[s], 1); mbar_init(&B.empty[s], 1); }
    mbar_init(&B.acc_full, 1); mbar_init(&B.acc_empty, 16);
    for (int g = 0; g < 2; ++g) { mbar_init(&B.staged[g], 4); mbar_init(&B.stage_free[g], 1); }
    fence_barrier_init();
    prefetch_map(&mdy); prefetch_map(&mxn); prefetch_map(&mwst); prefetch_map(&mwa); prefetch_map(&mwb); prefetch_map(&mh); prefetch_map(&mdab);
  }
  if (warp == 2) { tmem_alloc2(smem_u32(&B.tmem), 512); tmem_relinquish2(); }
  tc_fence_before();
  __syncthreads();
  cluster_sync();
  tc_fence_after();
  const uint32_t tmem = B.tmem;

  if (warp < 4) setmaxnreg_dec<56>();
  if (warp == 0) {
    if (lane == 0) {
      const CUtensorMap* mab = leader ? &mwa : &mwb;
      int st = 0;
      for (int q = 0; q < nitem; ++q) {
        const int i = q / NPAIR, pr = q % NPAIR, row = tile_of(i) * ROWS;
        for (int kb = 0; kb < NKB; ++kb, ++st) {
          const int s = st % NST;
          if (st >= NST) mbar_wait(&B.empty[s], ((st / NST) - 1) & 1);
          if (leader) mbar_expect_tx(&B.full[s], 2 * STAGE);
          const uint32_t base = su + O_ST + s * STAGE;
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            tma_load_2d_2sm(base + S_DY + h * 8192, &mdy, &B.full[s], kb * 64, row + h * 64);
            tma_load_2d_2sm(base + S_XN + h * 8192, &mxn, &B.full[s], kb * 64, row + h * 64);
          }
          tma_load_2d_2sm(base + S_WAB, mab, &B.full[s], kb * 64, pr * 128);             // Wa (leader) / Wb (peer) rows of the pair
          tma_load_2d_2sm(base + S_WS, &mwst, &B.full[s], kb * 64, pr * 128 + crank * 64); // Ws^T rows of chunk 2 p + crank
        }
      }
    }
  } else if (warp == 1) {
    if (leader) {
      int st = 0;
      for (int q = 0; q < nitem; ++q) {
        if (q >= 1) mbar_wait_cl(&B.acc_empty, (q - 1) & 1);
        for (int kb = 0; kb < NKB; ++kb, ++st) {
          const int sg = st % NST;
          mbar_wait(&B.full[sg], (st / NST) & 1);
          tc_fence_after();
          const uint32_t base = su + O_ST + sg * STAGE;
          const uint64_t ddy = desc_k128(base + S_DY), dxn = desc_k128(base + S_XN);
          const uint64_t dab = desc_k128(base + S_WAB), dws = desc_k128(base + S_WS);
          if (elect_one()) {
#pragma unroll
            for (int ks = 0; ks < 4; ++ks) {
              umma_ss2(tmem + T_DH, ddy + (uint64_t)(ks * 2), dws + (uint64_t)(ks * 2), IDESC_DH, (kb > 0 || ks > 0) ? 1u : 0u);
              umma_ss2(tmem + T_AB, dxn + (uint64_t)(ks * 2), dab + (uint64_t)(ks * 2), IDESC_AB, (kb > 0 || ks > 0) ? 1u : 0u);
            }
            tc_commit2_mc(&B.empty[sg], 3);
            if (kb == NKB - 1) tc_commit2_mc(&B.acc_full, 3);
          }
          __syncwarp();
        }
      }
    }
  } else if ((warp >= 4 && warp < 8) || warp >= 12) {
    setmaxnreg_inc<152>();
    const int g = warp >= 12 ? 1 : 0;                           // this warpgroup's chunk: 2 p + g
    const uint32_t lb = (uint32_t)(warp & 3) * 32, r = lb + lane, trow = tmem + (lb << 16);
    const uint32_t ob = su + O_OUT + g * 3 * KBT;
    for (int q = 0; q < nitem; ++q) {
      mbar_wait(&B.acc_full, q & 1);
      tc_fence_after();
#pragma unroll
      for (int t = 0; t < 2; ++t) {                             // units 32 t .. 32 t + 31 of the chunk
        uint32_t dh[32], av[32], bv[32];
        tmem_ld32(trow + T_DH + g * 64 + t * 32, dh);
        tmem_ld32(trow + T_AB + g * 64 + t * 32, av);
        tmem_ld32(trow + T_AB + 128 + g * 64 + t * 32, bv);
        tmem_wait_ld();
        if (t == 1) {                                           // the accumulator is no longer needed by this warp
          tc_fence_before();
          __syncwarp();
          if (lane == 0) mbar_arrive_remote_relaxed(&B.acc_empty, 0);
        }
        uint32_t hq[16], aq[16], bq[16];
#pragma unroll
        for (int k = 0; k < 16; ++k) gate_pair16(dh[2 * k], dh[2 * k + 1], av[2 * k], av[2 * k + 1], bv[2 * k], bv[2 * k + 1], hq[k], aq[k], bq[k]);
        if (t == 0 && q >= 1) mbar_wait(&B.stage_free[g], (q - 1) & 1);
#pragma unroll
        for (int qq = 0; qq < 4; ++qq) {
          const uint32_t off = sw128(r, t * 4 + qq);
          sts128(ob + off, make_uint4(hq[4 * qq], hq[4 * qq + 1], hq[4 * qq + 2], hq[4 * qq + 3]));
          sts128(ob + KBT + off, make_uint4(aq[4 * qq], aq[4 * qq + 1], aq[4 * qq + 2], aq[4 * qq + 3]));
          sts128(ob + 2 * KBT + off, make_uint4(bq[4 * qq], bq[4 * qq + 1], bq[4 * qq + 2], bq[4 * qq + 3]));
        }
      }
      fence_proxy_async();
      __syncwarp();
      if (lane == 0) mbar_arrive(&B.staged[g]);
    }
  } else if (warp == 8) {
    // stores: per item, warpgroup 0's chunk then warpgroup 1's
    if (lane == 0) {
      for (int q = 0; q < nitem; ++q) {
        const int i = q / NPAIR, pr = q % NPAIR;
#pragma unroll
        for (int g = 0; g < 2; ++g) {
          mbar_wait(&B.staged[g], q & 1);
          const uint32_t ob = su + O_OUT + g * 3 * KBT;
          const int j = 2 * pr + g;
          if (i < n_valid) {
            const int row = tile_of(i) * ROWS;
#pragma unroll
            for (int h = 0; h < 2; ++h) {
              if (store_h) tma_store_2d(&mh, ob + h * 8192, j * HS, row + h * 64);
              tma_store_2d(&mdab, ob + KBT + h * 8192, j * HS, row + h * 64);
              tma_store_2d(&mdab, ob + 2 * KBT + h * 8192, H_ + j * HS, row + h * 64);
            }
          }
          tma_store_commit();
        }
        asm volatile("cp.async.bulk.wait_group.read 1;" ::: "memory");   // warpgroup 0's group has been read
        mbar_arrive(&B.stage_free[0]);
        tma_store_wait_read0();
        mbar_arrive(&B.stage_free[1]);
      }
      tma_store_wait0();
    }
  }
  tc_fence_before();
  __syncthreads();
  cluster_sync();
  if (warp == 2) { tc_fence_after(); tmem_dealloc2(tmem, 512); }
}
