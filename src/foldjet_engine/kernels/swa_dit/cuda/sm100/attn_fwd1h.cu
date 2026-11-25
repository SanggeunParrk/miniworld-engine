// attn_fwd1h.cu — the SWA atom block's window attention forward on sm_100a for few items (small inference batches, e.g. A = 1): one
// head per item, the two warpgroups splitting its keys. Same math as attn_fwd3.cu (keys j with |i - j| <= 64 and j < seqused[n];
// O = softmax(q K^T scale) V in bf16, padding query rows 0; LSE = m + log l, 0 on rows without a valid key), P against the row maximum
// over both 128-key blocks, except the order of the sum l (block 0's sum + block 1's).
// Items = (sample, 128-query tile i0, head): twice attn_fwd3's count, a 40-KB stage (q | K | V over keys [i0 - 64, i0 + 192)).
// Warpgroup kb takes key block kb (columns 128 kb .. + 127 of the item's S) for all 128 query rows (thread = row = TMEM lane): row
// maximum and sum of its block, exchanged with the other warpgroup through shared memory; P of block kb over its S columns (as
// attn_fwd3), PV of block kb as soon as it is written. S is double-buffered in TMEM (item parity), so the next item's S MMAs need not
// wait for this item's O read-out. 32-column chunks outside a warp's windows are skipped, inside every row's window unmasked.
// TMEM per buffer b (256 columns at 256 b): S0 at 0, S1 at 128; P0 over S0 columns 0-63, P1 over S1 columns 0-63; O over S0 columns 96-127.
// SPDX-License-Identifier: Apache-2.0
#include "sm100.cuh"
using namespace s100;

constexpr int DH = 32, QM = 128, H = 4, HW = 64, KR = 256;
constexpr int TQ = QM * 64, TK = KR * 64;                                   // q tile 8 KB, K / V 256-row box 16 KB
constexpr int IST = TQ + 2 * TK;                                            // q | K | V = 40 KB
constexpr int O_ST = 0, O_X = 2 * IST, XS = 128 * 64, O_RX = O_X + XS;      // then O staging [128][64 B] SW64, row exchange
constexpr int O_BAR = O_RX + 2 * 2 * 128 * 4, SMEM_BYTES = O_BAR + 256;     // rmax[2][128], rsum[2][128]
static_assert(SMEM_BYTES <= 232448, "shared memory");
constexpr uint32_t I_S = idesc_bf16(128, 128), I_PV = idesc_bf16(128, DH, 0, 1);
constexpr float LOG2E = 1.4426950408889634f, LN2 = 0.6931471805599453f;

struct Bars {
  uint64_t full[2], empty[2], s_full[2], p_full[2], o_full[2], t_free[2];
  uint32_t tmem;
};

extern "C" __global__ void __launch_bounds__(384, 1)
swa_attn_fwd1h_sm100(const __grid_constant__ CUtensorMap mq, const __grid_constant__ CUtensorMap mk, const __grid_constant__ CUtensorMap mv,
                     const __grid_constant__ CUtensorMap mo, const int* __restrict__ SEQU, float* __restrict__ LSE, int S, int N, float scale,
                     unsigned long long* __restrict__ TRb) {   // (TRb: the AttnFwd2 launch signature; unused)
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  Bars& B = *reinterpret_cast<Bars*>(sm + O_BAR);
  float* rmax = reinterpret_cast<float*>(sm + O_RX);                       // [2 blocks][128 rows]
  float* rsum = rmax + 256;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int mt = S / QM;
  const int items = N * mt * H;
  const int my_items = (items > (int)blockIdx.x) ? (items - (int)blockIdx.x + (int)gridDim.x - 1) / (int)gridDim.x : 0;
  auto item_of = [&](int li, int& n, int& i0, int& h, int& sq) {
    const int wi = (int)blockIdx.x + li * (int)gridDim.x;
    h = wi % H; const int r = wi / H;
    i0 = (r % mt) * QM; n = r / mt;
    sq = __ldg(SEQU + n);
  };

  if (tid == 0) {
    for (int s = 0; s < 2; ++s) {
      mbar_init(&B.full[s], 1); mbar_init(&B.empty[s], 1); mbar_init(&B.s_full[s], 1); mbar_init(&B.p_full[s], 4);
      mbar_init(&B.o_full[s], 1); mbar_init(&B.t_free[s], 8);
    }
    fence_barrier_init();
  }
  if (warp == 2) { tmem_alloc(smem_u32(&B.tmem), 512); tmem_relinquish(); }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = B.tmem;
  pdl_launch();

  if (warp == 0) {
    // ------------------------------------------------------------------------------------------------ TMA producer: q, K, V of the head
    if (lane == 0) {
      pdl_wait();                                                          // q / K / V come from the previous kernel
      for (int li = 0; li < my_items; ++li) {
        int n, i0, h, sq; item_of(li, n, i0, h, sq);
        const int s = li & 1;
        if (li >= 2) mbar_wait(&B.empty[s], ((li >> 1) - 1) & 1);
        const uint32_t st = su + O_ST + s * IST;
        mbar_expect_tx(&B.full[s], TQ + 2 * TK);
        const int row = (n * H + h) * S;
        tma_load_2d(st, &mq, &B.full[s], 0, row + i0);
        tma_load_2d(st + TQ, &mk, &B.full[s], 0, row + i0 - HW);           // keys [i0 - 64, i0 + 192): rows outside the head are masked
        tma_load_2d(st + TQ + TK, &mv, &B.full[s], 0, row + i0 - HW);      //  (or zero-filled)
      }
    }
  } else if (warp == 1) {
    // ------------------------------------------------------------------------------------------------ MMA issuer
    for (int li = 0; li < my_items; ++li) {
      const int s = li & 1, b = li & 1;
      const uint32_t st = su + O_ST + s * IST, tb = tmem + 256 * b;
      mbar_wait(&B.full[s], (li >> 1) & 1);
      if (li >= 2) mbar_wait(&B.t_free[b], ((li >> 1) - 1) & 1);          // buffer b's O (item li - 2) has been read out
      tc_fence_after();
      const uint64_t dq = desc_sw64(st);
      if (elect_one()) {
#pragma unroll
        for (int kb = 0; kb < 2; ++kb) {
          const uint64_t dk = desc_sw64(st + TQ + kb * 128 * 64);
#pragma unroll
          for (int ks = 0; ks < DH / 16; ++ks) umma_ss(tb + 128 * kb, dq + (uint64_t)(ks * 2), dk + (uint64_t)(ks * 2), I_S, ks > 0 ? 1u : 0u);
        }
        tc_commit(&B.s_full[b]);
      }
      __syncwarp();
#pragma unroll
      for (int kb = 0; kb < 2; ++kb) {
        mbar_wait(&B.p_full[kb], li & 1);
        tc_fence_after();
        const uint64_t dv = desc_sw64(st + TQ + TK + kb * 128 * 64);      // MN-major: 8-key groups 512 B apart
        if (elect_one()) {
#pragma unroll
          for (int ks = 0; ks < 8; ++ks)
            umma_ts(tb + 96, tb + 128 * kb + ks * 8, dv + (uint64_t)(ks * 64), I_PV, (kb > 0 || ks > 0) ? 1u : 0u);
          if (kb == 1) { tc_commit(&B.o_full[b]); tc_commit(&B.empty[s]); }
        }
        __syncwarp();
      }
    }
  } else if (warp >= 4) {
    pdl_wait();                                                            // LSE / O stores
    // ------------------------------------------------------------------------------------------------ softmax: warpgroup kb = key block kb
    const int kb = (warp - 4) >> 2;
    const uint32_t lb = (uint32_t)(warp & 3) * 32, r = lb + lane;
    const float SC = scale * LOG2E;
    const uint32_t z[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    for (int li = 0; li < my_items; ++li) {
      int n, i0, h, sq; item_of(li, n, i0, h, sq);
      const int b = li & 1;
      const uint32_t trow = tmem + (lb << 16) + 256 * b + 128 * kb;          // this row's columns of block kb
      const int i = i0 + (int)r, j0 = i0 - HW;
      const bool qok = i < sq;
      // valid columns of this row over the 256 keys, then relative to block kb: [clo, chi]
      const int clo = max(i - HW - j0, -j0) - 128 * kb, chi = min(i + HW, sq - 1) - j0 - 128 * kb;
      mbar_wait(&B.s_full[b], (li >> 1) & 1);
      tc_fence_after();
      const int wlo = __reduce_min_sync(~0u, chi >= clo ? clo : KR), whi = __reduce_max_sync(~0u, chi >= clo ? chi : -KR);
      auto live = [&](int c32) { return 32 * c32 + 31 >= wlo && 32 * c32 <= whi; };
      const int flo = __reduce_max_sync(~0u, clo), fhi = __reduce_min_sync(~0u, chi);
      const bool allq = __all_sync(~0u, qok);
      auto full = [&](int c32) { return allq && 32 * c32 >= flo && 32 * c32 + 31 <= fhi; };
      // pass 1: this block's row maximum, then the row's over both blocks
      float mx = -INFINITY;
#pragma unroll 1
      for (int c4 = 0; c4 < 4; ++c4) {
        if (!live(c4)) continue;
        uint32_t v[32];
        tmem_ld32(trow + 32 * c4, v);
        tmem_wait_ld();
        if (full(c4)) {
          float m4[4] = {mx, -INFINITY, -INFINITY, -INFINITY};
#pragma unroll
          for (int k = 0; k < 32; ++k) m4[k & 3] = fmaxf(m4[k & 3], __uint_as_float(v[k]));
          mx = fmaxf(fmaxf(m4[0], m4[1]), fmaxf(m4[2], m4[3]));
        } else {
#pragma unroll
          for (int k = 0; k < 32; ++k) { const int c = 32 * c4 + k; if (c >= clo && c <= chi) mx = fmaxf(mx, qok ? __uint_as_float(v[k]) : 0.f); }
        }
      }
      rmax[kb * 128 + r] = mx;
      named_bar_sync(1, 256);
      mx = fmaxf(rmax[r], rmax[128 + r]);
      const float NM = mx == -INFINITY ? 0.f : -mx * SC;
      // pass 2: P = 2^(S scale log2 e - m) over this block's S columns (P chunk k at columns 16 k), then its PV
      float l = 0.f;
#pragma unroll 1
      for (int cq = 0; cq < 4; ++cq) {
        if (!live(cq)) { tmem_st16(trow + 16 * cq, z); continue; }
        uint32_t v[32];
        tmem_ld32(trow + 32 * cq, v);
        tmem_wait_ld();
        uint32_t pk[16];
        if (full(cq)) {
#pragma unroll
          for (int k = 0; k < 16; ++k) {
            const float p0 = ex2f(fmaf(__uint_as_float(v[2 * k]), SC, NM)), p1 = ex2f(fmaf(__uint_as_float(v[2 * k + 1]), SC, NM));
            l += p0 + p1;
            pk[k] = pack_bf16(p0, p1);
          }
        } else {
#pragma unroll
          for (int k = 0; k < 16; ++k) {
            const int c0 = 32 * cq + 2 * k;
            const float p0 = ex2f(c0 >= clo && c0 <= chi ? fmaf(qok ? __uint_as_float(v[2 * k]) : 0.f, SC, NM) : -INFINITY);
            const float p1 = ex2f(c0 + 1 >= clo && c0 + 1 <= chi ? fmaf(qok ? __uint_as_float(v[2 * k + 1]) : 0.f, SC, NM) : -INFINITY);
            l += p0 + p1;
            pk[k] = pack_bf16(p0, p1);
          }
        }
        tmem_st16(trow + 16 * cq, pk);
      }
      tmem_wait_st();
      tc_fence_before();
      __syncwarp();
      if (lane == 0) mbar_arrive(&B.p_full[kb]);
      rsum[kb * 128 + r] = l;
      named_bar_sync(1, 256);
      l = rsum[r] + rsum[128 + r];
      // ---- epilogue: O = acc / l (bf16; 0 on padding / empty rows), this warpgroup's 16 of the 32 columns; LSE by warpgroup 0
      mbar_wait(&B.o_full[b], (li >> 1) & 1);
      tc_fence_after();
      uint32_t ov[16];
      tmem_ld16(tmem + (lb << 16) + 256 * b + 96 + 16 * kb, ov);
      tmem_wait_ld();
      tc_fence_before();
      __syncwarp();
      if (lane == 0) mbar_arrive(&B.t_free[b]);
      const bool ok = qok && l > 0.f;
      const float inv = ok ? 1.f / l : 0.f;
      const bool lead = tid == 128;
      if (lead) tma_store_wait_read0();
      named_bar_sync(1, 256);
#pragma unroll
      for (int q = 0; q < 2; ++q)
        sts128(su + O_X + sw64(r, 2 * kb + q), make_uint4(pack_bf16(__uint_as_float(ov[8 * q]) * inv, __uint_as_float(ov[8 * q + 1]) * inv),
                                                          pack_bf16(__uint_as_float(ov[8 * q + 2]) * inv, __uint_as_float(ov[8 * q + 3]) * inv),
                                                          pack_bf16(__uint_as_float(ov[8 * q + 4]) * inv, __uint_as_float(ov[8 * q + 5]) * inv),
                                                          pack_bf16(__uint_as_float(ov[8 * q + 6]) * inv, __uint_as_float(ov[8 * q + 7]) * inv)));
      fence_proxy_async();
      named_bar_sync(1, 256);
      if (lead) { tma_store_2d(&mo, su + O_X, h * DH, n * S + i0); tma_store_commit(); }
      if (kb == 0 && i < S) LSE[((size_t)n * H + h) * S + i] = l > 0.f ? (mx * SC + __log2f(l)) * LN2 : 0.f;
    }
    if (tid == 128) tma_store_wait0();
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 2) { tc_fence_after(); tmem_dealloc(tmem, 512); }
}
