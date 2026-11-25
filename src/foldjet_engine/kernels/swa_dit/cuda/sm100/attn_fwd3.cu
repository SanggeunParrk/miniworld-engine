// attn_fwd3.cu — attn_fwd2.cu with (a) the two heads' P phases taking turns (ALT, CTAs with more than one item): while one warpgroup
// writes P, the other takes its row maximum, epilogue and waits (per-warp latency, not the MUFU, bounds these phases); (b) 32-column
// chunks inside the window of every row of a warp without masks, their TMEM loads one chunk ahead. Same values as attn_fwd2 (bitwise).
// attn_fwd2.cu — the SWA atom block's window attention forward on sm_100a, two 128-key blocks per 128-query tile:
// keys j with |i - j| <= 64 and j < seqused[n]; O = softmax(q K^T scale) V in bf16 (padding query rows 0), LSE = m + log l (natural log,
// 0 on rows without a valid key).
// Items = (sample, 128-query tile i0, head pair); the two compute warpgroups take the two heads (one MMA warp each). The window of a query
// tile is exactly [i0 - 64, i0 + 192) = two 128-key blocks, loaded with the tile as one 256-row box per head. Both S blocks are computed
// at the start of the item (TMEM holds them side by side), so each row takes its true maximum over both blocks before any exponential:
// P = 2^(S scale log2 e - m) needs no rescaling of O (P rounds to bf16 against the row maximum, not a running one -- same accuracy class
// as the Triton kernel, not the same bits). P overwrites the S columns it came from; O accumulates over both PV products.
// Tiles are dense 64-B rows, 64-B swizzled; the item stage (q | K | V of both heads) is double-buffered, so the next item loads under
// this one's math.
// TMEM per head w (256 columns at 256 w): S0 at 0, S1 at 128; P0 over S0 columns 0-63, P1 over S1 columns 0-63; O over S0 columns 96-127.
// SPDX-License-Identifier: Apache-2.0
#include "sm100.cuh"
using namespace s100;
#ifndef TRACE
#define TRACE 0
#endif
#define EV(k) do { if (TRACE && blockIdx.x == 0 && (threadIdx.x & 127) == 0 && li < 16) TRb[400 + 256 * w + 16 * li + (k)] = gtime(); } while (0)
#define EVW(k) do { if (TRACE && blockIdx.x == 0 && lane == 0 && li < 16) TRb[400 + 16 * li + (k)] = gtime(); } while (0)
#ifndef ALT
#define ALT 1
#endif
DEVI void named_bar_arrive(int id, int n) { asm volatile("bar.arrive %0, %1;" :: "r"(id), "r"(n) : "memory"); }
DEVI unsigned long long gtime() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }

constexpr int DH = 32, QM = 128, H = 4, HW = 64, KR = 256;
constexpr int TQ = QM * 64, TK = KR * 64;                                   // q tile 8 KB, K / V 256-row box 16 KB (per head)
constexpr int IST = 2 * TQ + 4 * TK;                                        // q0 | q1 | K0 | K1 | V0 | V1 = 80 KB
constexpr int O_ST = 0, O_X = 2 * IST, XS = 128 * 64;
constexpr int O_BAR = O_X + 2 * XS, SMEM_BYTES = O_BAR + 256;
static_assert(SMEM_BYTES <= 232448, "shared memory");
constexpr uint32_t I_S = idesc_bf16(128, 128), I_PV = idesc_bf16(128, DH, 0, 1);
constexpr float LOG2E = 1.4426950408889634f, LN2 = 0.6931471805599453f;

struct Bars {
  uint64_t full[2], empty[2], s_full[2], p_full[2][2], o_full[2], t_free[2];
  uint32_t tmem;
};
#ifndef HINT
#define HINT 0                             // 0: plain try_wait loop; else the suspend-time hint (ns)
#endif
DEVI void mbar_wait_h(uint64_t* b, uint32_t parity) {
  if (!HINT) { mbar_wait(b, parity); return; }
  uint32_t ok;
  do {
    asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2, %3; selp.u32 %0, 1, 0, p; }"
                 : "=r"(ok) : "r"(smem_u32(b)), "r"(parity), "r"((uint32_t)HINT) : "memory");
  } while (!ok);
}

extern "C" __global__ void __launch_bounds__(384, 1)
swa_attn_fwd3_sm100(const __grid_constant__ CUtensorMap mq, const __grid_constant__ CUtensorMap mk, const __grid_constant__ CUtensorMap mv,
                    const __grid_constant__ CUtensorMap mo, const int* __restrict__ SEQU, float* __restrict__ LSE, int S, int N, float scale, unsigned long long* __restrict__ TRb) {
  if (TRACE && threadIdx.x == 0) TRb[2 * blockIdx.x] = gtime();
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  Bars& B = *reinterpret_cast<Bars*>(sm + O_BAR);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int mt = S / QM;
  const int items = N * mt * 2;
  const int my_items = (items > (int)blockIdx.x) ? (items - (int)blockIdx.x + (int)gridDim.x - 1) / (int)gridDim.x : 0;
  auto item_of = [&](int li, int& n, int& i0, int& hp, int& sq) {
    const int wi = (int)blockIdx.x + li * (int)gridDim.x;
    hp = wi & 1; const int r = wi >> 1;
    i0 = (r % mt) * QM; n = r / mt;
    sq = __ldg(SEQU + n);
  };

  if (tid == 0) {
    for (int s = 0; s < 2; ++s) { mbar_init(&B.full[s], 1); mbar_init(&B.empty[s], 2); }
    for (int w = 0; w < 2; ++w) {
      mbar_init(&B.s_full[w], 1); mbar_init(&B.p_full[w][0], 4); mbar_init(&B.p_full[w][1], 4); mbar_init(&B.o_full[w], 1); mbar_init(&B.t_free[w], 4);
    }
    fence_barrier_init();
  }
  if (warp == 2) { tmem_alloc(smem_u32(&B.tmem), 512); tmem_relinquish(); }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = B.tmem;
  pdl_launch();
  if (TRACE && blockIdx.x == 0 && threadIdx.x == 0) TRb[399] = gtime();

  if (warp == 0) {
    // ------------------------------------------------------------------------------------------------ TMA producer: q, K, V (both heads)
    if (lane == 0) {
      pdl_wait();                                                          // q / K / V come from the previous kernel
      for (int li = 0; li < my_items; ++li) {
        int n, i0, hp, sq; item_of(li, n, i0, hp, sq);
        const int s = li & 1;
        if (li >= 2) mbar_wait_h(&B.empty[s], ((li >> 1) - 1) & 1);
        const uint32_t st = su + O_ST + s * IST;
        EVW(0);
        mbar_expect_tx(&B.full[s], 2 * TQ + 4 * TK);
        for (int w = 0; w < 2; ++w) {
          const int row = (n * H + 2 * hp + w) * S;
          tma_load_2d(st + w * TQ, &mq, &B.full[s], 0, row + i0);
          tma_load_2d(st + 2 * TQ + w * TK, &mk, &B.full[s], 0, row + i0 - HW);         // keys [i0 - 64, i0 + 192): rows outside the
          tma_load_2d(st + 2 * TQ + 2 * TK + w * TK, &mv, &B.full[s], 0, row + i0 - HW); //  head are masked (or zero-filled)
        }
      }
    }
  } else if (warp == 1 || warp == 2) {
    // ------------------------------------------------------------------------------------------------ MMA issuers (warp 1 + w: head 2 hp + w)
    const int w = warp - 1;
    const uint32_t tb = tmem + 256 * w;
    for (int li = 0; li < my_items; ++li) {
      const int s = li & 1;
      const uint32_t st = su + O_ST + s * IST;
      mbar_wait_h(&B.full[s], (li >> 1) & 1);
      if (w == 0) EVW(1);
      if (li >= 1) mbar_wait_h(&B.t_free[w], (li - 1) & 1);                 // the previous item's O has been read out
      tc_fence_after();
      const uint64_t dq = desc_sw64(st + w * TQ);
      if (elect_one()) {
#pragma unroll
        for (int kb = 0; kb < 2; ++kb) {
          const uint64_t dk = desc_sw64(st + 2 * TQ + w * TK + kb * 128 * 64);
#pragma unroll
          for (int ks = 0; ks < DH / 16; ++ks) umma_ss(tb + 128 * kb, dq + (uint64_t)(ks * 2), dk + (uint64_t)(ks * 2), I_S, ks > 0 ? 1u : 0u);
        }
        tc_commit(&B.s_full[w]);
      }
      __syncwarp();
#pragma unroll
      for (int kb = 0; kb < 2; ++kb) {
        mbar_wait_h(&B.p_full[w][kb], li & 1);
        tc_fence_after();
        const uint64_t dv = desc_sw64(st + 2 * TQ + 2 * TK + w * TK + kb * 128 * 64);    // MN-major: 8-key groups 512 B apart
        if (elect_one()) {
#pragma unroll
          for (int ks = 0; ks < 8; ++ks)
            umma_ts(tb + 96, tb + 128 * kb + ks * 8, dv + (uint64_t)(ks * 64), I_PV, (kb > 0 || ks > 0) ? 1u : 0u);
          if (kb == 1) { tc_commit(&B.o_full[w]); tc_commit(&B.empty[s]); }
        }
        __syncwarp();
      }
    }
  } else if (warp >= 4) {
    pdl_wait();                                                            // LSE / O stores
    // ------------------------------------------------------------------------------------------------ softmax of head 2 hp + w, one query row per thread
    const int w = (warp - 4) >> 2;
    const uint32_t lb = (uint32_t)(warp & 3) * 32, r = lb + lane, trow = tmem + (lb << 16) + 256 * w;
    const float SC = scale * LOG2E;
    const bool alt = ALT && my_items > 1;                                   // turns: head 0 of item li, head 1 of li, head 0 of li + 1, ...
    for (int li = 0; li < my_items; ++li) {
      int n, i0, hp, sq; item_of(li, n, i0, hp, sq);
      const int i = i0 + (int)r, j0 = i0 - HW;                               // block kb covers keys j0 + 128 kb .. + 127
      const bool qok = i < sq;
      // valid key columns of this row (over the 256 keys of both blocks): [clo, chi]
      const int clo = max(i - HW - j0, -j0), chi = min(i + HW, sq - 1) - j0;   // (padding query rows: scores 0, as Triton's zero q)
      mbar_wait_h(&B.s_full[w], li & 1);
      EV(2);
      tc_fence_after();
      // the warp's 32 rows need key columns [wlo, whi] only: 32-column chunks outside are neither loaded nor exponentiated
      const int wlo = __reduce_min_sync(~0u, chi >= clo ? clo : KR), whi = __reduce_max_sync(~0u, chi >= clo ? chi : -1);
      auto live = [&](int c32) { return 32 * c32 + 31 >= wlo && 32 * c32 <= whi; };
      // chunks valid on all 32 columns for every row of the warp (no padding rows): no masks
      const int flo = __reduce_max_sync(~0u, clo), fhi = __reduce_min_sync(~0u, chi);
      const bool allq = __all_sync(~0u, qok);
      auto full = [&](int c32) { return allq && 32 * c32 >= flo && 32 * c32 + 31 <= fhi; };
      // the live chunks [cl, ch] are contiguous. Each pass takes the edge chunks one at a time (masked) and sweeps the inner ones, all
      // full when mid holds (always, except at sequence ends / padding rows), with the next chunk's TMEM load in flight during the math.
      const int cl = wlo >> 5, ch = whi >> 5;
      const bool mid = ch - cl >= 2 && full(cl + 1) && full(ch - 1);
      auto one = [&](int c, auto&& proc) { uint32_t v[32]; tmem_ld32(trow + 32 * c, v); tmem_wait_ld(); proc(v, c); };
      auto pass = [&](auto&& pfull, auto&& pmask) {
        if (cl > ch) return;
        one(cl, pmask);
        if (mid) {
          uint32_t va[32], vb[32];
          tmem_ld32(trow + 32 * (cl + 1), va);
#pragma unroll 1
          for (int c = cl + 1; c < ch; c += 2) {
            tmem_wait_ld();
            if (c + 1 < ch) tmem_ld32(trow + 32 * (c + 1), vb);
            pfull(va, c);
            if (c + 1 >= ch) break;
            tmem_wait_ld();
            if (c + 2 < ch) tmem_ld32(trow + 32 * (c + 2), va);
            pfull(vb, c + 1);
          }
        } else {
#pragma unroll 1
          for (int c = cl + 1; c < ch; ++c) one(c, pmask);
        }
        if (ch > cl) one(ch, pmask);
      };
      // pass 1: the row maximum over both blocks
      float mx = -INFINITY;
      pass([&](uint32_t (&v)[32], int) {
        float m4[4] = {mx, -INFINITY, -INFINITY, -INFINITY};
#pragma unroll
        for (int k = 0; k < 32; ++k) m4[k & 3] = fmaxf(m4[k & 3], __uint_as_float(v[k]));
        mx = fmaxf(fmaxf(m4[0], m4[1]), fmaxf(m4[2], m4[3]));
      }, [&](uint32_t (&v)[32], int c4) {
#pragma unroll
        for (int k = 0; k < 32; ++k) { const int c = 32 * c4 + k; if (c >= clo && c <= chi) mx = fmaxf(mx, qok ? __uint_as_float(v[k]) : 0.f); }
      });
      EV(3);
      const float NM = mx == -INFINITY ? 0.f : -mx * SC;
      float l = 0.f;
      if (alt && (w == 1 || li > 0)) named_bar_sync(w == 0 ? 4 : 3, 256);  // the other head's exponentials are done
      // pass 2: P = 2^(S scale log2 e - m) over the S columns just read (P of chunk c at columns 128 (c / 4) + 16 (c % 4)), PV after
      // each block. Dead chunks get P = 0: those before the live range at once (their S columns are dead), those after it once the live
      // chunks have been read (their P columns lie in live S chunks).
      auto pcol = [&](int c) { return trow + 128 * (c >> 2) + 16 * (c & 3); };
      const uint32_t z[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
      auto pdone = [&](int kb) {
        tmem_wait_st();
        tc_fence_before();
        __syncwarp();
        if (lane == 0) mbar_arrive(&B.p_full[w][kb]);
        EV(4 + kb);
      };
#pragma unroll 1
      for (int c = 0; c < min(cl, 8); ++c) tmem_st16(pcol(c), z);
      if (cl >= 4) pdone(0);
      pass([&](uint32_t (&v)[32], int c4) {
        uint32_t pk[16];
#pragma unroll
        for (int k = 0; k < 16; ++k) {
          const float p0 = ex2f(fmaf(__uint_as_float(v[2 * k]), SC, NM)), p1 = ex2f(fmaf(__uint_as_float(v[2 * k + 1]), SC, NM));
          l += p0 + p1;
          pk[k] = pack_bf16(p0, p1);
        }
        tmem_st16(pcol(c4), pk);
        if (c4 == 3) pdone(0);
      }, [&](uint32_t (&v)[32], int c4) {
        uint32_t pk[16];
#pragma unroll
        for (int k = 0; k < 16; ++k) {
          const int c0 = 32 * c4 + 2 * k;
          const float p0 = ex2f(c0 >= clo && c0 <= chi ? fmaf(qok ? __uint_as_float(v[2 * k]) : 0.f, SC, NM) : -INFINITY);
          const float p1 = ex2f(c0 + 1 >= clo && c0 + 1 <= chi ? fmaf(qok ? __uint_as_float(v[2 * k + 1]) : 0.f, SC, NM) : -INFINITY);
          l += p0 + p1;
          pk[k] = pack_bf16(p0, p1);
        }
        tmem_st16(pcol(c4), pk);
        if (c4 == 3) pdone(0);
      });
#pragma unroll 1
      for (int c = max(cl, ch + 1); c < 8; ++c) tmem_st16(pcol(c), z);
      if (cl < 4 && ch < 3) pdone(0);
      pdone(1);
      if (alt && (w == 0 || li + 1 < my_items)) named_bar_arrive(w == 0 ? 3 : 4, 256);
      // ---- epilogue: O = acc / l (bf16; 0 on padding / empty rows), LSE = m ln 2 + log l
      mbar_wait_h(&B.o_full[w], li & 1);
      EV(6);
      tc_fence_after();
      uint32_t ov[32];
      tmem_ld32(trow + 96, ov);
      tmem_wait_ld();
      tc_fence_before();
      __syncwarp();
      if (lane == 0) mbar_arrive(&B.t_free[w]);
      const bool ok = qok && l > 0.f;
      const float inv = ok ? 1.f / l : 0.f;
      const uint32_t xs = su + O_X + w * XS;
      if (r == 0) tma_store_wait_read0();
      named_bar_sync(1 + w, 128);
#pragma unroll
      for (int q = 0; q < 4; ++q)
        sts128(xs + sw64(r, q), make_uint4(pack_bf16(__uint_as_float(ov[8 * q]) * inv, __uint_as_float(ov[8 * q + 1]) * inv),
                                           pack_bf16(__uint_as_float(ov[8 * q + 2]) * inv, __uint_as_float(ov[8 * q + 3]) * inv),
                                           pack_bf16(__uint_as_float(ov[8 * q + 4]) * inv, __uint_as_float(ov[8 * q + 5]) * inv),
                                           pack_bf16(__uint_as_float(ov[8 * q + 6]) * inv, __uint_as_float(ov[8 * q + 7]) * inv)));
      fence_proxy_async();
      named_bar_sync(1 + w, 128);
      const int h = 2 * hp + w;
      if (r == 0) { tma_store_2d(&mo, xs, h * DH, n * S + i0); tma_store_commit(); }
      EV(7);
      if (i < S) LSE[((size_t)n * H + h) * S + i] = l > 0.f ? (mx * SC + __log2f(l)) * LN2 : 0.f;
    }
    if (r == 0) tma_store_wait0();
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 2) { tc_fence_after(); tmem_dealloc(tmem, 512); }
  if (TRACE && threadIdx.x == 0) TRb[2 * blockIdx.x + 1] = gtime();
}
