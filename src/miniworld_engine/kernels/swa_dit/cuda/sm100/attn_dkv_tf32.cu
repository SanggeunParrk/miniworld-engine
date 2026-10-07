// attn_dkv_tf32.cu — the fp32 SWA atom block's window-attention backward, dK / dV pass, on sm_100a with TF32 tensor cores (tcgen05.mma
// kind::tf32, fp32 accumulation) and fp32 operands throughout (the fp32 counterpart of attn_dkv.cu; the Triton fp32 path ran bf16 operands):
// keys j < seqused[n], queries i with |i - j| <= 64 and i < seqused[n]
//   S^T = K q^T, dP^T = V dO^T (TMEM, fp32);  P^T = exp(S^T scale - LSE);  dS^T = P^T (dP^T - D)
//   dV = sum_blocks P^T dO;  dK = scale sum_blocks dS^T q
// Q / K / V head-major [N, H, S, 32] fp32 (the TF32 forward's saves), dO row-major [N S, 128] fp32 (oproj_bwd_tf32.cu: TF32-valued), LSE / D
// [N, H, S] fp32 (LSE: natural log of the sum of exp of the scaled scores).
// One head of 32 fp32 is one 128-B row, so every tile is a plain K-major 128-B-swizzled box [rows][32]; the MN-major B operands of
// dV += P^T dO and dK += dS^T q (dO / q with the queries as K) must sit in the 128-B swizzle with 32-B atoms for kind::tf32 (TMA "128a32",
// UMMA layout type 1; the plain swizzle multiplies to zeros), so dO and q come in twice (K-major for S^T / dP^T, MN-major for dV / dK).
// Items (sample, 128-key tile, head): the window's <= 8 32-query blocks [j0 - 64, j0 + 192) stream through a ring with their LSE / D (bulk
// copies). The two compute warpgroups take alternate blocks (one key row per thread); ONE MMA warp issues every MMA in block order, so the
// shared dK / dV accumulators sum in a fixed order (deterministic). Per block: S^T / dP^T (SS, M = 128 keys, N = 32, K = 32; double-buffered
// per warpgroup) -> P^T, dS^T (rounded to TF32) into TMEM -> dV += P^T dO, dK += dS^T q (TS, A from TMEM, M = 128, N = 32, K = 32).
// Epilogue: warpgroup 0 runs dK (scaled) through unrope + the head RMS backward (qkvg_bwd: dpk), warpgroup 1 rounds dV to TF32 (dpv); both
// are stored into dP [M, 4C] (columns C + 32 h / 2 C + 32 h), one 128-B row per thread
// via a shared-memory tile and a TMA store; the K / V slot is released after the item's last S^T / dP^T reads (two item slots).
// Shared memory: K | V item slots 2 x 32 KB | 7 block stages x 17 KB (q | dO | q MN | dO MN | LSE | D) | dK / dV staging 2 x 16 KB | barriers.
// TMEM: warpgroup w at 192 w: S^T[b] at + 64 b, dP^T[b] at + 64 b + 32, P^T at + 128, dS^T at + 160;  dK at 384, dV at 416.
// SPDX-License-Identifier: Apache-2.0
#include "sm100.cuh"
using namespace s100;

namespace tf {
__host__ __device__ constexpr uint32_t idesc(int M, int N, int a_mn = 0, int b_mn = 0) {
  return (1u << 4) | (2u << 7) | (2u << 10) | ((uint32_t)a_mn << 15) | ((uint32_t)b_mn << 16) | ((uint32_t)(N >> 3) << 17) |
         ((uint32_t)(M >> 4) << 24);
}
// MN-major operand in the 128-B swizzle with 32-B atoms: SBO 512, LBO = distance between 32-element MN atoms (one atom here)
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
DEVI uint32_t rna(float x) { uint32_t r; asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(x)); return r & 0xffffe000u; }
}  // namespace tf

#ifndef ST
#define ST 7                                                               // block stages
#endif
constexpr int BQ = 32, DH = 32, KM = 128, H = 4, HW = 64, QR = 2;
constexpr int TKV = KM * 128, IST = 2 * TKV;                               // K (or V) of the item: [128][32] fp32 = 16 KB; K | V
constexpr int TQ = BQ * 128;                                               // a [32][32] fp32 block tile: 4 KB
constexpr int BLS = 4 * TQ + 1024;                                         // q | dO | q MN | dO MN | LSE (128 B) | D (128 B), 1-KB aligned
constexpr int NIT = 64;                                                    // items of a CTA whose (n, tile, h, block range, seqused) sit in a table
constexpr int XS = KM * 128;                                               // output staging tile: [128 rows][32 fp32], SW128 (16 KB), one per warpgroup
constexpr int O_IT = 0, O_BL = QR * IST, O_X = O_BL + ST * BLS, O_TAB = O_X + 2 * XS, O_BAR = O_TAB + NIT * 16, SMEM_BYTES = O_BAR + 512;
static_assert(SMEM_BYTES <= 232448, "shared memory");
constexpr uint32_t T_DK = 384, T_DV = 416;
constexpr uint32_t I_S = tf::idesc(128, BQ), I_O = tf::idesc(128, DH, 0, 1);
constexpr float LOG2E = 1.4426950408889634f;

struct Bars {
  uint64_t it_full[QR], it_empty[QR], bl_full[ST], bl_empty[ST], s_full[2][2], s_free[2][2], p_full[2], p_free[2], acc_full, acc_free, tgo, tdone;
  uint32_t tmem;
};
DEVI void bulk_g2s(uint32_t dst, const void* src, uint32_t bytes, uint64_t* bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               :: "r"(dst), "l"(src), "r"(bytes), "r"(smem_u32(bar)) : "memory");
}

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

extern "C" __global__ void __launch_bounds__(384, 1)
swa_attn_dkv_tf32_sm100(const __grid_constant__ CUtensorMap mq, const __grid_constant__ CUtensorMap mqm,
                        const __grid_constant__ CUtensorMap mk, const __grid_constant__ CUtensorMap mv,
                        const __grid_constant__ CUtensorMap mdo, const __grid_constant__ CUtensorMap mdom,
                        const float* __restrict__ LSE, const float* __restrict__ DD, const int* __restrict__ SEQU,
                        const float* __restrict__ PKS, const float* __restrict__ COS, const float* __restrict__ SIN,
                        const __grid_constant__ CUtensorMap mdp, int S, int N, int Bn, float scale, float eps) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  Bars& B = *reinterpret_cast<Bars*>(sm + O_BAR);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int mt = S / KM;
  const int items = N * mt * H;
  const int my_items = (items > (int)blockIdx.x) ? (items - (int)blockIdx.x + (int)gridDim.x - 1) / (int)gridDim.x : 0;
  // item li of this CTA: wi = blockIdx + li grid = ((n mt + m) H + h); its query blocks [lo, hi) (one masked block when none is valid)
  auto item_raw = [&](int li, int& n, int& j0, int& h, int& lo, int& hi, int& sq) {
    const int wi = (int)blockIdx.x + li * (int)gridDim.x;
    h = wi % H; const int r = wi / H;
    j0 = (r % mt) * KM; n = r / mt;
    sq = __ldg(SEQU + n);
    lo = max(j0 - HW, 0) / BQ;
    hi = (min(sq, j0 + KM + HW) + BQ - 1) / BQ;
    if (hi <= lo) hi = lo + 1;
  };

  int4* itab = reinterpret_cast<int4*>(sm + O_TAB);                    // (n, 4 j0 + h, 65536 lo + hi, seqused) of item li < NIT: the
  auto item_of = [&](int li, int& n, int& j0, int& h, int& lo, int& hi, int& sq) {    // look-ahead of the MMA warp must not wait on a global load
    if (li >= NIT) { item_raw(li, n, j0, h, lo, hi, sq); return; }
    const int4 e = itab[li];
    n = e.x; j0 = e.y >> 2; h = e.y & 3; lo = e.z >> 16; hi = e.z & 0xffff; sq = e.w;
  };
  if (tid < my_items && tid < NIT) {
    int n, j0, h, lo, hi, sq; item_raw(tid, n, j0, h, lo, hi, sq);
    itab[tid] = make_int4(n, (j0 << 2) | h, (lo << 16) | hi, sq);
  }

  if (tid == 0) {
    for (int s = 0; s < QR; ++s) { mbar_init(&B.it_full[s], 1); mbar_init(&B.it_empty[s], 1); }
    for (int s = 0; s < ST; ++s) { mbar_init(&B.bl_full[s], 1); mbar_init(&B.bl_empty[s], 1); }
    for (int w = 0; w < 2; ++w) {
      for (int b = 0; b < 2; ++b) { mbar_init(&B.s_full[w][b], 1); mbar_init(&B.s_free[w][b], 4); }
      mbar_init(&B.p_full[w], 4); mbar_init(&B.p_free[w], 1);
    }
    mbar_init(&B.acc_full, 1); mbar_init(&B.acc_free, 8);
    mbar_init(&B.tgo, 1); mbar_init(&B.tdone, 1);
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
        int n, j0, h, lo, hi, sq; item_of(li, n, j0, h, lo, hi, sq);
        const int is = li % QR, krow = (n * H + h) * S + j0;
        if (li >= QR) mbar_wait(&B.it_empty[is], ((li / QR) - 1) & 1);
        mbar_expect_tx(&B.it_full[is], IST);
        tma_load_2d(su + O_IT + is * IST, &mk, &B.it_full[is], 0, krow);
        tma_load_2d(su + O_IT + is * IST + TKV, &mv, &B.it_full[is], 0, krow);
        for (int qb = lo; qb < hi; ++qb, ++g) {
          const int s = g % ST;
          if (g >= ST) mbar_wait(&B.bl_empty[s], ((g / ST) - 1) & 1);
          const uint32_t st = su + O_BL + s * BLS;
          const int qrow = (n * H + h) * S + qb * BQ, orow = n * S + qb * BQ;
          mbar_expect_tx(&B.bl_full[s], 4 * TQ + 2 * BQ * 4);
          tma_load_2d(st, &mq, &B.bl_full[s], 0, qrow);
          tma_load_2d(st + TQ, &mdo, &B.bl_full[s], h * DH, orow);
          tma_load_2d(st + 2 * TQ, &mqm, &B.bl_full[s], 0, qrow);
          tma_load_2d(st + 3 * TQ, &mdom, &B.bl_full[s], h * DH, orow);
          bulk_g2s(st + 4 * TQ, LSE + qrow, BQ * 4, &B.bl_full[s]);
          bulk_g2s(st + 4 * TQ + 128, DD + qrow, BQ * 4, &B.bl_full[s]);
        }
      }
    }
  } else if (warp == 1) {
    // ------------------------------------------------------------------------------------------------ MMA issuer (block G -> warpgroup G & 1)
    auto issue_s = [&](int G, int li, int qb, int lo, int hi) {            // S^T / dP^T of block G into its warpgroup's buffer (G >> 1) & 1
      const int s = G % ST, is = li % QR, w = G & 1, gw = G >> 1, bs = gw & 1;
      if (qb == lo) mbar_wait(&B.it_full[is], (li / QR) & 1);
      mbar_wait(&B.bl_full[s], (G / ST) & 1);
      if (gw >= 2) mbar_wait(&B.s_free[w][bs], ((gw >> 1) - 1) & 1);       // warpgroup w has read S^T / dP^T of its block gw - 2
      tc_fence_after();
      const uint32_t kv = su + O_IT + is * IST, st = su + O_BL + s * BLS, d = tmem + 192 * w + 64 * bs;
      if (elect_one()) {
#pragma unroll
        for (int k = 0; k < 4; ++k) tf::mma_ss(d, desc_k128(kv) + (uint64_t)(k * 2), desc_k128(st) + (uint64_t)(k * 2), I_S, k > 0 ? 1u : 0u);
#pragma unroll
        for (int k = 0; k < 4; ++k)
          tf::mma_ss(d + 32, desc_k128(kv + TKV) + (uint64_t)(k * 2), desc_k128(st + TQ) + (uint64_t)(k * 2), I_S, k > 0 ? 1u : 0u);
        tc_commit(&B.s_full[w][bs]);
        if (qb == hi - 1) tc_commit(&B.it_empty[is]);                      // the item's last read of K / V
      }
      __syncwarp();
    };
    int nli = 0, nqb = 0, nlo = 0, nhi = 0;                                // the block after the current one (S^T / dP^T look-ahead)
    auto adv = [&]() {
      if (++nqb == nhi) { if (++nli < my_items) { int n, j0, h, sq; item_of(nli, n, j0, h, nlo, nhi, sq); nqb = nlo; } }
    };
    if (my_items > 0) {
      int n, j0, h, sq; item_of(0, n, j0, h, nlo, nhi, sq); nqb = nlo;
      issue_s(0, 0, nqb, nlo, nhi);
      adv();
    }
    int G = 0;
    for (int li = 0; li < my_items; ++li) {
      int n, j0, h, lo, hi, sq; item_of(li, n, j0, h, lo, hi, sq);
      for (int qb = lo; qb < hi; ++qb, ++G) {
        if (nli < my_items) { issue_s(G + 1, nli, nqb, nlo, nhi); adv(); }
        const int s = G % ST, w = G & 1, gw = G >> 1;
        mbar_wait(&B.p_full[w], gw & 1);                                   // P^T / dS^T of block G are in TMEM
        if (qb == lo && li >= 1) mbar_wait(&B.acc_free, (li - 1) & 1);     // the previous item's dK / dV have been read out
        tc_fence_after();
        const uint32_t st = su + O_BL + s * BLS, tp = tmem + 192 * w + 128;
        if (elect_one()) {
#pragma unroll
          for (int k = 0; k < 4; ++k)                                      // K = 32 queries: 8 per step (1 KB of the MN-major tile)
            tf::mma_ts(tmem + T_DV, tp + 8 * k, tf::desc_mn32b(st + 3 * TQ, TQ) + (uint64_t)((k * 1024) >> 4), I_O, (qb > lo || k > 0) ? 1u : 0u);
#pragma unroll
          for (int k = 0; k < 4; ++k)
            tf::mma_ts(tmem + T_DK, tp + 32 + 8 * k, tf::desc_mn32b(st + 2 * TQ, TQ) + (uint64_t)((k * 1024) >> 4), I_O, (qb > lo || k > 0) ? 1u : 0u);
          tc_commit(&B.p_free[w]);
          tc_commit(&B.bl_empty[s]);
          if (qb == hi - 1) tc_commit(&B.acc_full);
        }
        __syncwarp();
      }
    }
  } else if (warp == 2 || warp == 3) {
    // ------------------------------------------------------------------------------------------------ tail warps: the head-wise qkvg backward tail
    // The compute warpgroups stage the raw (scaled) dK of an item in a shared-memory tile and arrive on tgo; these 64 threads (warp 2 has only
    // allocated TMEM, warp 3 does nothing else) run unrope + the per-head RMS backward on it -- the loads of the raw projection and of cos / sin
    // (latency that stalled the in-order MMA warp's pipeline when the compute warpgroups did this) are theirs -- and store the tile by TMA.
    pdl_wait();
    const int t = (warp - 2) * 32 + lane;
    for (int li = 0; li < my_items; ++li) {
      int n, j0, h, lo, hi, sq; item_of(li, n, j0, h, lo, hi, sq);
      mbar_wait(&B.tgo, li & 1);
      for (int rr = t; rr < 128; rr += 64) {
        float d[32];
#pragma unroll
        for (int q = 0; q < 8; ++q) {
          const uint4 x = lds128((su + O_X) + sw128((uint32_t)rr, (uint32_t)q));
          d[4 * q] = __uint_as_float(x.x); d[4 * q + 1] = __uint_as_float(x.y); d[4 * q + 2] = __uint_as_float(x.z); d[4 * q + 3] = __uint_as_float(x.w);
        }
        const size_t m = (size_t)n * S + j0 + rr;
        head_bwd(d, PKS + m * 128 + 32 * h, COS + ((size_t)(n % Bn) * S + j0 + rr) * 16, SIN + ((size_t)(n % Bn) * S + j0 + rr) * 16, eps);
#pragma unroll
        for (int q = 0; q < 8; ++q)
          sts128((su + O_X) + sw128((uint32_t)rr, (uint32_t)q), make_uint4(__float_as_uint(d[4 * q]), __float_as_uint(d[4 * q + 1]), __float_as_uint(d[4 * q + 2]), __float_as_uint(d[4 * q + 3])));
      }
      fence_proxy_async();
      named_bar_sync(6, 64);
      if (t == 0) {
        tma_store_2d(&mdp, (su + O_X), 128 + 32 * h, n * S + j0);
        tma_store_commit();
        tma_store_wait_read0();                                            // the tile is free again: the next item's epilogue may stage into it
        mbar_arrive(&B.tdone);
      }
    }
    if (t == 0) tma_store_wait0();
  } else if (warp >= 4) {
    // ------------------------------------------------------------------------------------------------ P^T / dS^T, one key row per thread
    const int w = (warp - 4) >> 2, lq = warp & 3, r = lq * 32 + lane;
    const uint32_t trow = tmem + ((uint32_t)(lq * 32) << 16);
    const float SC = scale * LOG2E;
    int G = 0;
    for (int li = 0; li < my_items; ++li) {
      int n, j0, h, lo, hi, sq; item_of(li, n, j0, h, lo, hi, sq);
      const int j = j0 + r;
      const bool kok = j < sq;
      for (int qb = lo; qb < hi; ++qb, ++G) {
        if ((G & 1) != w) continue;
        const int s = G % ST, gw = G >> 1, bs = gw & 1;
        const uint32_t st = su + O_BL + s * BLS;
        mbar_wait(&B.s_full[w][bs], (gw >> 1) & 1);
        tc_fence_after();
        uint32_t sv[32], dv[32];
        tmem_ld32(trow + 192 * w + 64 * bs, sv);
        tmem_ld32(trow + 192 * w + 64 * bs + 32, dv);
        tmem_wait_ld();
        tc_fence_before();
        __syncwarp();
        if (lane == 0) mbar_arrive(&B.s_free[w][bs]);
        const int i0 = qb * BQ;
        const int clo = max(j - HW - i0, 0), chi = kok ? min(min(j + HW, sq - 1) - i0, BQ - 1) : -1;
        const float4* l4 = reinterpret_cast<const float4*>(sm + (st - su) + 4 * TQ);
        const float4* d4 = reinterpret_cast<const float4*>(sm + (st - su) + 4 * TQ + 128);
#pragma unroll
        for (int q = 0; q < 8; ++q) {                                      // 4 queries per step: LSE / D as broadcast 16-B loads
          const float4 lv = l4[q], dd = d4[q];
          const float la[4] = {lv.x, lv.y, lv.z, lv.w}, da[4] = {dd.x, dd.y, dd.z, dd.w};
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const int c = 4 * q + e;
            const bool ok = c >= clo && c <= chi;
            const float p = ok ? ex2f(__uint_as_float(sv[c]) * SC - la[e] * LOG2E) : 0.f;
            const float ds = ok ? p * (__uint_as_float(dv[c]) - da[e]) : 0.f;
            sv[c] = tf::rna(p); dv[c] = tf::rna(ds);
          }
        }
        if (gw >= 1) mbar_wait(&B.p_free[w], (gw - 1) & 1);                // the dV / dK MMAs of this warpgroup's previous block are done
        tc_fence_after();
        const uint32_t tp = trow + 192 * w + 128;
        tmem_st16(tp, *reinterpret_cast<uint32_t(*)[16]>(sv));
        tmem_st16(tp + 16, *reinterpret_cast<uint32_t(*)[16]>(sv + 16));
        tmem_st16(tp + 32, *reinterpret_cast<uint32_t(*)[16]>(dv));
        tmem_st16(tp + 48, *reinterpret_cast<uint32_t(*)[16]>(dv + 16));
        tmem_wait_st();
        tc_fence_before();
        __syncwarp();
        if (lane == 0) mbar_arrive(&B.p_full[w]);
      }
      // ---- the item's dK (warpgroup 0, scaled) / dV (warpgroup 1, rounded): key row j, 32 fp32 = one 128-B row
      mbar_wait(&B.acc_full, li & 1);
      tc_fence_after();
      uint32_t v[32];
      tmem_ld32(trow + (w ? T_DV : T_DK), v);
      tmem_wait_ld();
      tc_fence_before();
      __syncwarp();
      if (lane == 0) mbar_arrive(&B.acc_free);
      // dV (warpgroup 1) is rounded and stored at once (dpv). dK (warpgroup 0) is scaled and staged raw: the tail warps run its head-wise tail
      // (unrope + head RMS backward -> dpk) and store it (dP [M, 4C] = [dpq | dpk | dpv | dg]: dpk at column C + 32 h, dpv at 2 C + 32 h)
      const uint32_t xs = su + O_X + w * XS;
      if (w == 0) { if (li >= 1) mbar_wait(&B.tdone, (li - 1) & 1); }      // the tail warps have read and stored the previous item's tile
      else if (r == 0) tma_store_wait_read0();
      if (w == 1) named_bar_sync(2, 128);
#pragma unroll
      for (int q = 0; q < 8; ++q) {
        uint32_t o[4];
#pragma unroll
        for (int e = 0; e < 4; ++e) o[e] = w ? tf::rna(__uint_as_float(v[4 * q + e])) : __float_as_uint(__uint_as_float(v[4 * q + e]) * scale);
        sts128(xs + sw128((uint32_t)r, (uint32_t)q), make_uint4(o[0], o[1], o[2], o[3]));
      }
      if (w) {
        fence_proxy_async();
        named_bar_sync(2, 128);
        if (r == 0) { tma_store_2d(&mdp, xs, 256 + 32 * h, n * S + j0); tma_store_commit(); }
      } else {
        named_bar_sync(1, 128);
        if (r == 0) mbar_arrive(&B.tgo);
      }
    }
  }
  if (warp >= 4 && (warp & 3) == 0 && lane == 0) tma_store_wait0();
  tc_fence_before();
  __syncthreads();
  if (warp == 2) { tc_fence_after(); tmem_dealloc(tmem, 512); }
}
