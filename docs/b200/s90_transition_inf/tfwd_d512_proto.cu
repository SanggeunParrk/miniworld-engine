// tfwd_d512.cu (S90 PROTOTYPE) -- the Transition forward at D = 512 (hidden HID = n D) as ONE fused sm_100a kernel on a CTA PAIR that SPLITS the output columns: both CTAs of a cluster work on the SAME 128-row tile,
// CTA q owns output columns 256 q .. 256 q + 255 (256 TMEM columns of the squeeze accumulator). Per 64-unit hidden chunk j (h never leaves the SM):
//   expand   each CTA: [a | b] of ITS 32 units (units 64 j + 32 q ..) = xn [Wa_j,q ; Wb_j,q]^T   M 128, N 64, K 512 (eight K-blocks of the resident xn tile), cta_group::1, 2 x 64 TMEM columns (double-buffered)
//   SwiGLU   the CTA's 4 warps: h[:, 32 units] bf16 into its own h slot AND by DSMEM st.async into the peer's (chunk-major 16-B packets, complete_tx on the peer's barrier): both h slots hold the 64-unit chunk
//   squeeze  out[:, 256 q ..] += h Ws_j,q^T   M 128, N 256, K 64 (this CTA's 256 Ws rows of the chunk, 32 KB), accumulated in 256 TMEM columns over the 16 chunks of the tile
//   epilogue out = bf16(acc + x) by 64-column blocks (residual by TMA into the Ws stage, TMA store)
// xn comes from the LayerNorm launch (transition_ln_w); the K16 fold is not in this prototype. Shared memory (bytes): xn tile 131072, B ring 3 x 8192, Ws stage 32768, h 2 x 16384, barriers.
// SPDX-License-Identifier: Apache-2.0
#include "sm100.cuh"
#include "tsx_store.cuh"
using namespace s100;

#ifndef HID
#define HID 1024
#endif
#ifndef NBS_
#define NBS_ 3
#endif
constexpr int D_ = 512, H_ = HID, HS = 64, NCH = H_ / HS, ROWS = 128, NKB = D_ / 64, NBS = NBS_;
constexpr int KBT = ROWS * 128;                                // 16 KB: one K-block of the 128-row tile
constexpr int O_X = 0, O_B = NKB * KBT, O_WS = O_B + NBS * 8192, O_H = O_WS + 32768, O_BAR = O_H + 2 * KBT;
constexpr int SMEM_BYTES = O_BAR + 1024;
static_assert(SMEM_BYTES <= 232448, "shared memory budget");
constexpr uint32_t T_AB = 0, T_OUT = 128;                      // 2 x 64 [a | b] columns, 256 squeeze columns
constexpr uint32_t IDESC_EX = idesc_bf16(128, 64), IDESC_SQ = idesc_bf16(128, 256);

struct Bars {
  uint64_t xfull, xdead, bfull[NBS], bempty[NBS], abfull[2], abempty[2], hfull[2], hempty[2], hpeer[2], wsfull, wsempty, outfull, outempty, epidone, xres[2];
  uint32_t tmem;
};

extern "C" __global__ void __launch_bounds__(384, 1)
transition_fwd_d512_sm100(const __grid_constant__ CUtensorMap mxn, const __grid_constant__ CUtensorMap mwa, const __grid_constant__ CUtensorMap mwb,
                          const __grid_constant__ CUtensorMap mws, const __grid_constant__ CUtensorMap mx, const __grid_constant__ CUtensorMap mout, int tiles) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  Bars& B = *reinterpret_cast<Bars*>(sm + O_BAR);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int crank = (int)cluster_rank(), pair = blockIdx.x >> 1, npairs = gridDim.x >> 1;
  const int n_local = (tiles > pair) ? (tiles - pair + npairs - 1) / npairs : 0;
  auto tile_of = [&](int i) { return pair + i * npairs; };

  if (tid == 0) {
    mbar_init(&B.xfull, 1); mbar_init(&B.xdead, 1);
    for (int s = 0; s < NBS; ++s) { mbar_init(&B.bfull[s], 1); mbar_init(&B.bempty[s], 1); }
    for (int s = 0; s < 2; ++s) { mbar_init(&B.abfull[s], 1); mbar_init(&B.abempty[s], 4); mbar_init(&B.hfull[s], 4); mbar_init(&B.hempty[s], 1); mbar_init(&B.hpeer[s], 1); mbar_init(&B.xres[s], 1); }
    mbar_init(&B.wsfull, 1); mbar_init(&B.wsempty, 1); mbar_init(&B.outfull, 1); mbar_init(&B.outempty, 1); mbar_init(&B.epidone, 1);
    fence_barrier_init();
    prefetch_map(&mxn); prefetch_map(&mwa); prefetch_map(&mwb); prefetch_map(&mws); prefetch_map(&mx); prefetch_map(&mout);
  }
  if (warp == 2) { tmem_alloc(smem_u32(&B.tmem), 512); tmem_relinquish(); }
  tc_fence_before();
  __syncthreads();
  cluster_sync_relaxed();
  tc_fence_after();
  const uint32_t tmem = B.tmem;

  if (warp == 0) {
    // ------------------------------------------------------------------------------------------ producers: lane 0 the xn tile and the expand weights, lane 1 the Ws stages
    if (lane == 0) {
      int st = 0;
      for (int i = 0; i < n_local; ++i) {
        const int row = tile_of(i) * ROWS;
        if (i >= 1) mbar_wait(&B.xdead, (i - 1) & 1);
#ifdef TBX
        mbar_arrive(&B.xfull);
#else
        mbar_expect_tx(&B.xfull, NKB * KBT);
        for (int kb = 0; kb < NKB; ++kb)
          for (int h = 0; h < 2; ++h) tma_load_2d(su + O_X + kb * KBT + h * 8192, &mxn, &B.xfull, kb * 64, row + h * 64);
#endif
        for (int j = 0; j < NCH; ++j)
          for (int kb = 0; kb < NKB; ++kb, ++st) {
            const int s = st % NBS;
            if (st >= NBS) mbar_wait(&B.bempty[s], ((st / NBS) - 1) & 1);
#ifndef TBB
            mbar_expect_tx(&B.bfull[s], 8192);
#endif
            const uint32_t dst = su + O_B + s * 8192;
#ifdef TBB
            mbar_arrive(&B.bfull[s]);
#else
            tma_load_2d(dst, &mwa, &B.bfull[s], kb * 64, j * HS + 32 * crank);
            tma_load_2d(dst + 4096, &mwb, &B.bfull[s], kb * 64, j * HS + 32 * crank);
#endif
          }
      }
    } else if (lane == 1) {
      int gcs = 0;
      for (int i = 0; i < n_local; ++i)
        for (int j = 0; j < NCH; ++j, ++gcs) {
          if (j == 0 && i >= 1) mbar_wait(&B.epidone, (i - 1) & 1);
          if (gcs >= 1) mbar_wait(&B.wsempty, (gcs - 1) & 1);
#ifndef TBWS
          mbar_expect_tx(&B.wsfull, 32768);
#endif
#ifdef TBWS
          mbar_arrive(&B.wsfull);
#else
          for (int b = 0; b < 4; ++b) tma_load_2d(su + O_WS + b * 8192, &mws, &B.wsfull, j * HS, crank * 256 + b * 64);
#endif
        }
    }
  } else if (warp == 1) {
    // ------------------------------------------------------------------------------------------ MMA issue: expand of chunk j, then the squeeze of chunk j - 1
    int st = 0;
    auto squeeze = [&](int i, int jj) {
      const int gcs = i * NCH + jj, s = gcs & 1;
      mbar_wait_cl(&B.hfull[s], (gcs >> 1) & 1);
      mbar_wait(&B.wsfull, gcs & 1);
      if (jj == 0 && i >= 1) mbar_wait(&B.outempty, (i - 1) & 1);
      tc_fence_after();
      if (elect_one()) {
        const uint64_t ha = desc_k128(su + O_H + s * KBT), wb = desc_k128(su + O_WS);
#pragma unroll
        for (int ks = 0; ks < 4; ++ks) umma_ss(tmem + T_OUT, ha + (uint64_t)(ks * 2), wb + (uint64_t)(ks * 2), IDESC_SQ, (jj > 0 || ks > 0) ? 1u : 0u);
        tc_commit(&B.hempty[s]);
        tc_commit(&B.wsempty);
        if (jj == NCH - 1) tc_commit(&B.outfull);
      }
      __syncwarp();
    };
    for (int i = 0; i < n_local; ++i) {
      mbar_wait(&B.xfull, i & 1);

      for (int j = 0; j < NCH; ++j) {
        const int gc = i * NCH + j, b = gc & 1;
        if (gc >= 2) mbar_wait(&B.abempty[b], ((gc >> 1) - 1) & 1);
        for (int kb = 0; kb < NKB; ++kb, ++st) {
          const int s = st % NBS;
          mbar_wait(&B.bfull[s], (st / NBS) & 1);
          tc_fence_after();
          if (elect_one()) {
            const uint64_t da = desc_k128(su + O_X + kb * KBT), db = desc_k128(su + O_B + s * 8192);
#pragma unroll
            for (int ks = 0; ks < 4; ++ks) umma_ss(tmem + T_AB + b * 64, da + (uint64_t)(ks * 2), db + (uint64_t)(ks * 2), IDESC_EX, (kb > 0 || ks > 0) ? 1u : 0u);
            tc_commit(&B.bempty[s]);
            if (kb == NKB - 1) { tc_commit(&B.abfull[b]); if (j == NCH - 1) tc_commit(&B.xdead); }
          }
          __syncwarp();
        }
        if (j >= 1) squeeze(i, j - 1);
      }
      squeeze(i, NCH - 1);
    }
  } else if (warp == 2) {
    // ------------------------------------------------------------------------------------------ forwards "my h slot is free" to the peer (its SwiGLU pushes into my slot)
    if (lane == 1) {
      const int total = n_local * NCH;
      for (int gc = 0; gc < total; ++gc) { mbar_wait(&B.hempty[gc & 1], (gc >> 1) & 1); mbar_arrive_remote_cta(&B.hpeer[gc & 1], (uint32_t)(crank ^ 1)); }
    }
  } else if (warp >= 4 && warp < 8) {
    // ------------------------------------------------------------------------------------------ SwiGLU of the CTA's 32 units, h into both CTAs' slots
    const uint32_t lb = (uint32_t)(warp & 3) * 32, r = lb + lane, trow = tmem + (lb << 16);
    for (int i = 0; i < n_local; ++i)
      for (int j = 0; j < NCH; ++j) {
        const int gc = i * NCH + j, b = gc & 1, s = b;
        mbar_wait(&B.abfull[b], (gc >> 1) & 1);
        tc_fence_after();
        uint32_t av[32], bv[32];
        tmem_ld32(trow + T_AB + b * 64, av);
        tmem_ld32(trow + T_AB + b * 64 + 32, bv);
        tmem_wait_ld();
        tc_fence_before();
        __syncwarp();
        if (lane == 0) mbar_arrive(&B.abempty[b]);
        uint32_t hp[16];
#pragma unroll
        for (int k = 0; k < 16; ++k) {
          const float a0 = __uint_as_float(av[2 * k]), a1 = __uint_as_float(av[2 * k + 1]);
          const float b0 = __uint_as_float(bv[2 * k]), b1 = __uint_as_float(bv[2 * k + 1]);
#ifdef TBSW
          hp[k] = pack_bf16(a0 * b0, a1 * b1);
#else
          hp[k] = pack_bf16(a0 * sigmoid_kit(a0) * b0, a1 * sigmoid_kit(a1) * b1);
#endif
        }
        if (gc >= 2) { mbar_wait(&B.hempty[s], ((gc >> 1) - 1) & 1); mbar_wait_cl(&B.hpeer[s], ((gc >> 1) - 1) & 1); }
        if (warp == 4 && lane == 0) mbar_expect_tx(&B.hfull[s], 8192);   // + the 8 KB the peer pushes into this slot
        else if (lane == 0) mbar_arrive(&B.hfull[s]);
#pragma unroll
        for (int c = 0; c < 4; ++c) {
          const uint32_t addr = su + O_H + s * KBT + sw128(r, 4 * crank + c);
          const uint4 v = make_uint4(hp[4 * c], hp[4 * c + 1], hp[4 * c + 2], hp[4 * c + 3]);
          sts128(addr, v);
          st_async_v4(addr, smem_u32(&B.hfull[s]), (uint32_t)(crank ^ 1), v.x, v.y, v.z, v.w);
        }
        fence_proxy_async();
      }
  } else if (warp >= 8 && warp < 12) {
    // ------------------------------------------------------------------------------------------ epilogue of the CTA's 256 output columns
    const uint32_t lb = (uint32_t)(warp & 3) * 32, r = lb + lane, trow = tmem + (lb << 16);
    const bool lead = (warp & 3) == 0 && lane == 0;
    for (int i = 0; i < n_local; ++i) {
      const int row0 = tile_of(i) * ROWS;
      mbar_wait_cl(&B.outfull, i & 1);
#ifdef TBEPI
      tc_fence_after(); tc_fence_before(); if (lead) { mbar_arrive(&B.outempty); mbar_arrive(&B.epidone); } continue;
#endif
      if (lead) { mbar_expect_tx(&B.xres[0], KBT); for (int h = 0; h < 2; ++h) tma_load_2d(su + O_WS + h * 8192, &mx, &B.xres[0], crank * 256, row0 + h * 64); }
      tc_fence_after();
      for (int blk = 0; blk < 4; ++blk) {
        const int sl = blk & 1;
        const uint32_t xb = su + O_WS + sl * KBT;
        if (lead && blk + 1 < 4) {
          mbar_expect_tx(&B.xres[sl ^ 1], KBT);
          for (int h = 0; h < 2; ++h) tma_load_2d(su + O_WS + (sl ^ 1) * KBT + h * 8192, &mx, &B.xres[sl ^ 1], crank * 256 + (blk + 1) * 64, row0 + h * 64);
        }
        uint32_t acc[64];
        tmem_ld32(trow + T_OUT + blk * 64, *reinterpret_cast<uint32_t(*)[32]>(acc));
        tmem_ld32(trow + T_OUT + blk * 64 + 32, *reinterpret_cast<uint32_t(*)[32]>(acc + 32));
        tmem_wait_ld();
        mbar_wait(&B.xres[sl], ((i * 4 + blk) >> 1) & 1);
#pragma unroll
        for (int q = 0; q < 8; ++q) {
          const uint32_t ad = xb + sw128(r, q);
          const uint4 xv = lds128(ad);
          const uint32_t xw[4] = {xv.x, xv.y, xv.z, xv.w};
          uint32_t o[4];
#pragma unroll
          for (int e = 0; e < 4; ++e) o[e] = pack_bf16(bf16lo(xw[e]) + __uint_as_float(acc[q * 8 + 2 * e]), bf16hi(xw[e]) + __uint_as_float(acc[q * 8 + 2 * e + 1]));
          sts128(ad, make_uint4(o[0], o[1], o[2], o[3]));
        }
        fence_proxy_async();
        tc_fence_before();
        named_bar_sync(1, 128);
        if (lead && blk == 3) mbar_arrive(&B.outempty);
        if (lead) {
          for (int h = 0; h < 2; ++h) tma_store_2d(&mout, xb + h * 8192, crank * 256 + blk * 64, row0 + h * 64);
          tma_store_commit();
          tma_store_wait_read0();
        }
        named_bar_sync(1, 128);
      }
      tc_fence_before();
      if (lead) { tma_store_wait0(); mbar_arrive(&B.epidone); }
    }
  }
  tc_fence_before();
  __syncthreads();
  cluster_sync_relaxed();
  if (warp == 2) { tc_fence_after(); tmem_dealloc(tmem, 512); }
}

extern "C" __device__ int transition_smem_bytes = SMEM_BYTES;
