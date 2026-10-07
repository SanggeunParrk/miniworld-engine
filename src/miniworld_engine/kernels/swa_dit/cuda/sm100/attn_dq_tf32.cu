// attn_dq_tf32.cu — the fp32 SWA atom block's window-attention backward, dQ pass, on sm_100a with TF32 tensor cores (tcgen05.mma kind::tf32,
// fp32 accumulation) and fp32 operands throughout (the fp32 counterpart of attn_dq.cu; the Triton fp32 path ran bf16 operands): keys j with
// |i - j| <= 64 and j < seqused[n], queries i < seqused[n]
//   S = q K^T, dP = dO V^T (TMEM, fp32);  P = exp(S scale - LSE);  dS = P (dP - D);  dQ = scale sum_blocks dS K
// Q / K / V head-major [N, H, S, 32] fp32 (the TF32 forward's saves: TF32-valued), dO row-major [N S, 128] fp32 (oproj_bwd_tf32.cu: TF32-valued),
// LSE / D [N, H, S] fp32 (LSE: natural log of the sum of exp of the scaled scores), dQ out head-major fp32.
//
// Why a pass of its own: dQ = dS K needs dS with the queries as rows; the key-major pass (attn_dkv_tf32.cu) holds dS^T with the keys as TMEM
// lanes. Recomputing S / dP here, query-major, keeps every operand in a layout that is measured on B200 (K-major SS for S / dP, TS with an
// MN-major B for dQ) and each query tile's dQ is complete inside one item: deterministic, no atomics, no fp32 dQ buffer.
// Items (sample, 128-query tile, head); the window's <= 8 32-key blocks [i0 - 64, i0 + 192) stream through a ring of 12 KB stages
// (K, V K-major and K MN-major: the MN-major B operand of dQ must sit in the 128-B swizzle with 32-B atoms for kind::tf32, TMA "128a32",
// UMMA layout type 1: the plain swizzle multiplies to zeros). The two compute warpgroups take alternate blocks (one query row per thread);
// two MMA warps issue the MMAs: one the S / dP of every block (SS), one the dQ accumulation (TS)
// in block order, so the shared dQ accumulator sums in a fixed order. Per block: S / dP (SS, M = 128 queries,
// N = 32 keys, K = 32; double-buffered per warpgroup) -> dS (rounded to TF32) into TMEM -> dQ += dS K (TS, M = 128, N = 32, K = 32 keys).
// Epilogue: a warpgroup of its own (12-15, one query row per thread) reads the item's dQ out of its double-buffered TMEM accumulator, scales it,
// runs the head-wise qkvg backward (unrope + head RMS backward: dpq) and stores it into dP [M, 4C] at column 32 h by TMA, while the row
// threads and the MMA warps already run the next item.
// Warps: 0 TMA producer, 1 S / dP issuer, 2 TMEM allocator, 3 dQ issuer, 4-11 row threads, 12-15 epilogue (512 threads).
// Shared memory: q | dO item slots 2 x 32 KB | 12 block stages x 12 KB | dQ staging 16 KB | barriers. TMEM: warpgroup w at 160 w: S[b] at + 64 b, dP[b] at
// + 64 b + 32, dS at + 128; dQ of item li at 320 + 32 (li & 1).
// SPDX-License-Identifier: Apache-2.0
#include "sm100.cuh"
using namespace s100;

namespace tf {
__host__ __device__ constexpr uint32_t idesc(int M, int N, int a_mn = 0, int b_mn = 0) {
  return (1u << 4) | (2u << 7) | (2u << 10) | ((uint32_t)a_mn << 15) | ((uint32_t)b_mn << 16) | ((uint32_t)(N >> 3) << 17) |
         ((uint32_t)(M >> 4) << 24);
}
// MN-major operand in the 128-B swizzle with 32-B atoms (TMA "128a32", UMMA layout type 1): SBO 512, LBO = bytes between 32-element MN atoms
DEVI uint64_t desc_mn32b(uint32_t saddr, uint32_t lbo) {
  return (uint64_t)((saddr >> 4) & 0x3FFFu) | ((uint64_t)((lbo >> 4) & 0x3FFFu) << 16) | ((uint64_t)(512 >> 4) << 32) |
         ((uint64_t)1 << 46) | ((uint64_t)1 << 61);
}
DEVI void mma_ss(uint32_t d, uint64_t a, uint64_t b, uint32_t id, uint32_t acc) {
  asm volatile("{ .reg .pred p; setp.ne.b32 p, %4, 0; tcgen05.mma.cta_group::1.kind::tf32 [%0], %1, %2, %3, p; }"
               :: "r"(d), "l"(a), "l"(b), "r"(id), "r"(acc) : "memory");
}
DEVI void mma_ts(uint32_t d, uint32_t a, uint64_t b, uint32_t id, uint32_t acc) {
  asm volatile("{ .reg .pred p; setp.ne.b32 p, %4, 0; tcgen05.mma.cta_group::1.kind::tf32 [%0], [%1], %2, %3, p; }"
               :: "r"(d), "r"(a), "l"(b), "r"(id), "r"(acc) : "memory");
}
// cvt.rna.tf32.f32 leaves the 13 low bits zero (checked on B200)
DEVI uint32_t rna(float x) { uint32_t r; asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(x)); return r; }
}  // namespace tf

#ifndef ST
#define ST 12                                                              // block stages
#endif
constexpr int BK = 32, DH = 32, QM = 128, H = 4, HW = 64, QR = 2;
constexpr int TQI = QM * 128, IST = 2 * TQI;                               // q (or dO) of the item: [128][32] fp32 = 16 KB; q | dO
constexpr int TK = BK * 128, BLS = 3 * TK;                                 // a [32 keys][32] tile: 4 KB; K | V | K MN-major
constexpr int NIT = 64;                                                    // items of a CTA whose (n, tile, h, block range, seqused) sit in a table
constexpr int XS = QM * 128;                                               // dQ staging tile: [128 rows][32 fp32], SW128 (16 KB)
constexpr int O_IT = 0, O_BL = QR * IST, O_X = O_BL + ST * BLS, O_TAB = O_X + XS, O_BAR = O_TAB + NIT * 16, SMEM_BYTES = O_BAR + 512;
static_assert(SMEM_BYTES <= 232448, "shared memory");
constexpr uint32_t T_DQ = 320;
constexpr uint32_t I_S = tf::idesc(128, BK), I_DQ = tf::idesc(128, DH, 0, 1);
constexpr float LOG2E = 1.4426950408889634f;

struct Bars {
  uint64_t it_full[QR], it_empty[QR], bl_full[ST], bl_empty[ST], s_full[2][2], s_free[2][2], d_full[2], d_free[2], q_full[2], q_free[2];
  uint32_t tmem;
};

// the head-wise tail of the qkvg backward for one (row, head) held by one thread (the Triton _swa_qkvg_bwd_fp32_kernel's dP):
//   dp = head_rms_bwd(p, unrope(d)),  head_rms_bwd(p, du) = r (du - yh mean_D(du yh)),  yh = p r,  r = 1 / sqrt(mean_D(p^2) + eps),
//   unrope = the transpose of the RoPE rotation: dx1 = dy1 c + dy2 s, dx2 = dy2 c - dy1 s (halves d < 16 | d >= 16)
// p: this row's raw projection (32 fp32 at P), cs / sn: the atom's cos / sin (16 fp32 each); d[32] in place -> TF32-rounded dp
DEVI void head_bwd(float (&d)[32], const float* __restrict__ P, const float* __restrict__ cs, const float* __restrict__ sn, float eps) {
  float u[32];
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const float4 c4 = *reinterpret_cast<const float4*>(cs + 4 * k), s4 = *reinterpret_cast<const float4*>(sn + 4 * k);
    const float c[4] = {c4.x, c4.y, c4.z, c4.w}, s[4] = {s4.x, s4.y, s4.z, s4.w};
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const float a = d[4 * k + e], b = d[16 + 4 * k + e];
      u[4 * k + e] = a * c[e] + b * s[e];
      u[16 + 4 * k + e] = b * c[e] - a * s[e];
    }
  }
  float p[32], ss = 0.f;
#pragma unroll
  for (int k = 0; k < 8; ++k) {
    const float4 q4 = *reinterpret_cast<const float4*>(P + 4 * k);
    p[4 * k] = q4.x; p[4 * k + 1] = q4.y; p[4 * k + 2] = q4.z; p[4 * k + 3] = q4.w;
  }
#pragma unroll
  for (int e = 0; e < 32; ++e) ss = fmaf(p[e], p[e], ss);
  const float r = 1.f / sqrtf(ss * (1.f / 32.f) + eps);
  float t = 0.f;
#pragma unroll
  for (int e = 0; e < 32; ++e) t = fmaf(u[e], p[e] * r, t);
  t *= (1.f / 32.f);
#pragma unroll
  for (int e = 0; e < 32; ++e) d[e] = __uint_as_float(tf::rna(r * (u[e] - p[e] * r * t)));
}

extern "C" __global__ void __launch_bounds__(512, 1)
swa_attn_dq_tf32_sm100(const __grid_constant__ CUtensorMap mq, const __grid_constant__ CUtensorMap mdo,
                       const __grid_constant__ CUtensorMap mk, const __grid_constant__ CUtensorMap mv,
                       const __grid_constant__ CUtensorMap mkm, const float* __restrict__ LSE, const float* __restrict__ DD,
                       const int* __restrict__ SEQU, const float* __restrict__ PQS, const float* __restrict__ COS,
                       const float* __restrict__ SIN, const __grid_constant__ CUtensorMap mdp, int S, int N, int Bn, float scale,
                       float eps) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  Bars& B = *reinterpret_cast<Bars*>(sm + O_BAR);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int mt = S / QM;
  const int items = N * mt * H;
  const int my_items = (items > (int)blockIdx.x) ? (items - (int)blockIdx.x + (int)gridDim.x - 1) / (int)gridDim.x : 0;
  // item li of this CTA: wi = blockIdx + li grid = ((n mt + m) H + h); its key blocks [lo, hi) (one masked block when none is valid)
  auto item_raw = [&](int li, int& n, int& i0, int& h, int& lo, int& hi, int& sq) {
    const int wi = (int)blockIdx.x + li * (int)gridDim.x;
    h = wi % H; const int r = wi / H;
    i0 = (r % mt) * QM; n = r / mt;
    sq = __ldg(SEQU + n);
    lo = max(i0 - HW, 0) / BK;
    hi = (min(sq, i0 + QM + HW) + BK - 1) / BK;
    if (hi <= lo) hi = lo + 1;
  };

  int4* itab = reinterpret_cast<int4*>(sm + O_TAB);                    // (n, 4 i0 + h, 65536 lo + hi, seqused) of item li < NIT: the
  auto item_of = [&](int li, int& n, int& i0, int& h, int& lo, int& hi, int& sq) {    // look-ahead of the MMA warp must not wait on a global load
    if (li >= NIT) { item_raw(li, n, i0, h, lo, hi, sq); return; }
    const int4 e = itab[li];
    n = e.x; i0 = e.y >> 2; h = e.y & 3; lo = e.z >> 16; hi = e.z & 0xffff; sq = e.w;
  };
  if (tid < my_items && tid < NIT) {
    int n, i0, h, lo, hi, sq; item_raw(tid, n, i0, h, lo, hi, sq);
    itab[tid] = make_int4(n, (i0 << 2) | h, (lo << 16) | hi, sq);
  }

  if (tid == 0) {
    for (int s = 0; s < QR; ++s) { mbar_init(&B.it_full[s], 1); mbar_init(&B.it_empty[s], 1); }
    for (int s = 0; s < ST; ++s) { mbar_init(&B.bl_full[s], 1); mbar_init(&B.bl_empty[s], 2); /* 2: the S / dP issuer and the dQ issuer */ }
    for (int w = 0; w < 2; ++w) {
      for (int b = 0; b < 2; ++b) { mbar_init(&B.s_full[w][b], 1); mbar_init(&B.s_free[w][b], 4); }
      mbar_init(&B.d_full[w], 4); mbar_init(&B.d_free[w], 1);
    }
    for (int b = 0; b < 2; ++b) { mbar_init(&B.q_full[b], 1); mbar_init(&B.q_free[b], 4); }          // 4: the epilogue warps
    fence_barrier_init();
  }
  if (warp == 2) { tmem_alloc(smem_u32(&B.tmem), 512); tmem_relinquish(); }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = B.tmem;
  pdl_launch();

  if (warp == 0) {
    // ------------------------------------------------------------------------------------------------ TMA producer
    if (lane == 0) {
      pdl_wait();                                                          // q / k / v / dO / lse / D come from the previous kernels
      int g = 0;
      for (int li = 0; li < my_items; ++li) {
        int n, i0, h, lo, hi, sq; item_of(li, n, i0, h, lo, hi, sq);
        const int is = li % QR;
        if (li >= QR) mbar_wait(&B.it_empty[is], ((li / QR) - 1) & 1);
        mbar_expect_tx(&B.it_full[is], IST);
        tma_load_2d(su + O_IT + is * IST, &mq, &B.it_full[is], 0, (n * H + h) * S + i0);
        tma_load_2d(su + O_IT + is * IST + TQI, &mdo, &B.it_full[is], h * DH, n * S + i0);
        for (int kb = lo; kb < hi; ++kb, ++g) {
          const int s = g % ST;
          if (g >= ST) mbar_wait(&B.bl_empty[s], ((g / ST) - 1) & 1);
          const uint32_t st = su + O_BL + s * BLS;
          const int krow = (n * H + h) * S + kb * BK;
          mbar_expect_tx(&B.bl_full[s], BLS);
          tma_load_2d(st, &mk, &B.bl_full[s], 0, krow);
          tma_load_2d(st + TK, &mv, &B.bl_full[s], 0, krow);
          tma_load_2d(st + 2 * TK, &mkm, &B.bl_full[s], 0, krow);
        }
      }
    }
  } else if (warp == 1) {
    // ------------------------------------------------------------------------------------------------ MMA issuer (block G -> warpgroup G & 1)
    auto issue_s = [&](int G, int li, int kb, int lo, int hi) {            // S / dP of block G into its warpgroup's buffer (G >> 1) & 1
      const int s = G % ST, is = li % QR, w = G & 1, gw = G >> 1, bs = gw & 1;
      if (kb == lo) mbar_wait(&B.it_full[is], (li / QR) & 1);
      mbar_wait(&B.bl_full[s], (G / ST) & 1);
      if (gw >= 2) mbar_wait(&B.s_free[w][bs], ((gw >> 1) - 1) & 1);       // warpgroup w has read S / dP of its block gw - 2
      tc_fence_after();
      const uint32_t it = su + O_IT + is * IST, st = su + O_BL + s * BLS, d = tmem + 160 * w + 64 * bs;
      if (elect_one()) {
#pragma unroll
        for (int k = 0; k < 4; ++k) tf::mma_ss(d, desc_k128(it) + (uint64_t)(k * 2), desc_k128(st) + (uint64_t)(k * 2), I_S, k > 0 ? 1u : 0u);
#pragma unroll
        for (int k = 0; k < 4; ++k)
          tf::mma_ss(d + 32, desc_k128(it + TQI) + (uint64_t)(k * 2), desc_k128(st + TK) + (uint64_t)(k * 2), I_S, k > 0 ? 1u : 0u);
        tc_commit(&B.s_full[w][bs]);
        tc_commit(&B.bl_empty[s]);
        if (kb == hi - 1) tc_commit(&B.it_empty[is]);                      // the item's last read of q / dO
      }
      __syncwarp();
    };
    int G = 0;
    for (int li = 0; li < my_items; ++li) {
      int n, i0, h, lo, hi, sq; item_of(li, n, i0, h, lo, hi, sq);
      for (int kb = lo; kb < hi; ++kb, ++G) issue_s(G, li, kb, lo, hi);
    }
  } else if (warp == 3) {
    // ------------------------------------------------------------------------------------------------ MMA issuer of dQ (block G -> warpgroup G & 1)
    // a warp of its own: the S / dP issuer's waits and the dQ issue no longer queue behind each other
    int G = 0;
    for (int li = 0; li < my_items; ++li) {
      int n, i0, h, lo, hi, sq; item_of(li, n, i0, h, lo, hi, sq);
      for (int kb = lo; kb < hi; ++kb, ++G) {
        const int s = G % ST, w = G & 1, gw = G >> 1;
        mbar_wait(&B.d_full[w], gw & 1);                                   // dS of block G is in TMEM
        if (kb == lo && li >= 2) mbar_wait(&B.q_free[li & 1], ((li >> 1) - 1) & 1);   // item li - 2's dQ has been read out
        tc_fence_after();
        const uint32_t st = su + O_BL + s * BLS;
        if (elect_one()) {
#pragma unroll
          for (int k = 0; k < 4; ++k)                                      // K = 32 keys: 8 per step (1 KB of the MN-major K tile)
            tf::mma_ts(tmem + T_DQ + 32 * (li & 1), tmem + 160 * w + 128 + 8 * k, tf::desc_mn32b(st + 2 * TK, TK) + (uint64_t)((k * 1024) >> 4), I_DQ,
                       (kb > lo || k > 0) ? 1u : 0u);
          tc_commit(&B.d_free[w]);
          tc_commit(&B.bl_empty[s]);
          if (kb == hi - 1) tc_commit(&B.q_full[li & 1]);
        }
        __syncwarp();
      }
    }
  } else if (warp >= 12) {
    // ------------------------------------------------------------------------------------------------ epilogue warpgroup (warps 12-15)
    // dQ of an item sits in a TMEM buffer of its own (double-buffered): this warpgroup reads it out, scales it, runs the head-wise qkvg-backward
    // tail (unrope + per-head RMS backward -> dpq) and stores it by TMA, while the row threads and the MMA warps already work on the next item.
    // Thread = query row r = TMEM lane (the quarter of warp & 3).
    pdl_wait();                                                            // the raw projection, cos / sin come from earlier kernels
    const int lq = warp & 3, r = lq * 32 + lane, et = tid - 384;
    const uint32_t trow = tmem + ((uint32_t)(lq * 32) << 16);
    for (int li = 0; li < my_items; ++li) {
      int n, i0, h, lo, hi, sq; item_of(li, n, i0, h, lo, hi, sq);
      mbar_wait(&B.q_full[li & 1], (li >> 1) & 1);
      tc_fence_after();
      uint32_t v[32];
      tmem_ld32(trow + T_DQ + 32 * (li & 1), v);
      tmem_wait_ld();
      tc_fence_before();
      __syncwarp();
      if (lane == 0) mbar_arrive(&B.q_free[li & 1]);
      float d[32];
#pragma unroll
      for (int e = 0; e < 32; ++e) d[e] = __uint_as_float(v[e]) * scale;
      const size_t m = (size_t)n * S + i0 + r;
      head_bwd(d, PQS + m * 128 + 32 * h, COS + ((size_t)(n % Bn) * S + i0 + r) * 16, SIN + ((size_t)(n % Bn) * S + i0 + r) * 16, eps);
      if (et == 0 && li >= 1) tma_store_wait_read0();                      // the previous item's tile has been read by the TMA engine
      named_bar_sync(1, 128);
#pragma unroll
      for (int q = 0; q < 8; ++q)
        sts128((su + O_X) + sw128((uint32_t)r, (uint32_t)q), make_uint4(__float_as_uint(d[4 * q]), __float_as_uint(d[4 * q + 1]), __float_as_uint(d[4 * q + 2]), __float_as_uint(d[4 * q + 3])));
      fence_proxy_async();
      named_bar_sync(1, 128);
      if (et == 0) {                                                       // dP [M, 4C] = [dpq | dpk | dpv | dg]: dpq at column 32 h
        tma_store_2d(&mdp, su + O_X, 0 + 32 * h, n * S + i0);
        tma_store_commit();
      }
    }
    if (et == 0) tma_store_wait0();
  } else if (warp >= 4 && warp < 12) {
    // ------------------------------------------------------------------------------------------------ dS, one query row per thread
    const int w = (warp - 4) >> 2, lq = warp & 3, r = lq * 32 + lane;
    const uint32_t trow = tmem + ((uint32_t)(lq * 32) << 16);
    const float SC = scale * LOG2E;
    int G = 0;
    for (int li = 0; li < my_items; ++li) {
      int n, i0, h, lo, hi, sq; item_of(li, n, i0, h, lo, hi, sq);
      const int i = i0 + r;
      const bool qok = i < sq;
      const size_t rowi = (size_t)(n * H + h) * S + i;
      pdl_wait();                                                          // lse / D are read straight from global memory
      const float l2 = qok ? __ldg(LSE + rowi) * LOG2E : 0.f, dd = qok ? __ldg(DD + rowi) : 0.f;
      for (int kb = lo; kb < hi; ++kb, ++G) {
        if ((G & 1) != w) continue;
        const int gw = G >> 1, bs = gw & 1;
        mbar_wait(&B.s_full[w][bs], (gw >> 1) & 1);
        tc_fence_after();
        uint32_t sv[32], dv[32];
        tmem_ld32(trow + 160 * w + 64 * bs, sv);
        tmem_ld32(trow + 160 * w + 64 * bs + 32, dv);
        tmem_wait_ld();
        tc_fence_before();
        __syncwarp();
        if (lane == 0) mbar_arrive(&B.s_free[w][bs]);
        const int j0 = kb * BK;
        const int clo = max(i - HW - j0, 0), chi = qok ? min(min(i + HW, sq - 1) - j0, BK - 1) : -1;
        const unsigned wd = (unsigned)max(chi - clo + 1, 0);                // width of the visible columns (0: none)
#pragma unroll
        for (int c = 0; c < 32; ++c) {
          const bool ok = (unsigned)(c - clo) < wd;
          const float p = ok ? ex2f(__uint_as_float(sv[c]) * SC - l2) : 0.f;
          dv[c] = tf::rna(ok ? p * (__uint_as_float(dv[c]) - dd) : 0.f);
        }
        if (gw >= 1) mbar_wait(&B.d_free[w], (gw - 1) & 1);                // the dQ MMA of this warpgroup's previous block is done
        tc_fence_after();
        const uint32_t td = trow + 160 * w + 128;
        tmem_st16(td, *reinterpret_cast<uint32_t(*)[16]>(dv));
        tmem_st16(td + 16, *reinterpret_cast<uint32_t(*)[16]>(dv + 16));
        tmem_wait_st();
        tc_fence_before();
        __syncwarp();
        if (lane == 0) mbar_arrive(&B.d_full[w]);
      }
    }
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 2) { tc_fence_after(); tmem_dealloc(tmem, 512); }
}
