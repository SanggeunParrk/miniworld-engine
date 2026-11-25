// k3w.cu -- TriMul output side on B200 (tcgen05) for D = 256 / 384 / 512: both output GEMMs on the raw operands with the
// LayerNorms folded into the weights, and the whole output epilogue in the same kernel:
//   p = rs_o (t^T Wp'^T - mu_o sp) + ep        Wp' = bf16(Wp g_o) [D, H], sp = rowsum(Wp'), ep = Wp b_o
//   g = rs_i (x Wg'^T  - mu_i sg) + eg         Wg' = bf16(Wg g_i) [D, D], sg = rowsum(Wg'), eg = Wg b_i
//   y = bf16(x + bf16(sigmoid(g) p ds))
// t is the channel-major contraction output [H, M] (H = 2D bidirectional, D one direction): it is the MN-major A operand as
// stored, so nothing is transposed or normalised in memory. mu / rs are per-token statistics (mu_o / rs_o over the H channels
// of t, mu_i / rs_i over the D channels of x).
//
// Work item = 128 tokens x 128 output channels (p and g accumulators, 2 x 128 TMEM columns, double buffered). K streams through
// a ring of 32 KB stages (A 16 KB + B 16 KB): H / 64 stages of (t block, Wp' block), then D / 64 of (x block, Wg' block).
// 128 + 32 EPW threads: warp 0 TMA (stages), warp 1 MMA, warp 2 TMEM alloc, warp 3 TMA (residual x tile); warps 4.. epilogue
// (thread = token; EPW = 8: two warps per TMEM lane quarter, each half of the item's channels -- 2026-11-17, the epilogue's sigmoid /
// fold arithmetic was latency-bound at one warp per SM sub-partition).
// Schedule (2026-11-17): with many token tiles per CTA ("grouped") a CTA runs every channel item of a token tile back to back, so the
// second item's t / x K-blocks are L2 hits of the first's on the same die (before: items of one tile on neighbouring CTAs, which on
// the two-die B200 could each fetch the tile from HBM).
#include "sm100.cuh"
#include "tmap.h"
#include <cstdlib>
#include <type_traits>

using namespace sm100;

#ifdef K3W_TL
// timeline probe (-DK3W_TL): %globaltimer marks per CTA (ns, comparable across SMs), read by k3w_tl()
static __device__ unsigned long long g_k3tl[160][32];
#define K3TL(slot) { unsigned long long t_; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t_)); g_k3tl[blockIdx.x][slot] = t_; }
#else
#define K3TL(slot) {}
#endif
#define K3T2(i_, e_) {}
namespace k3w {

#ifndef K3W_GREG
#define K3W_GREG 0   // 1: inference epilogue with the residual and y through registers / global (measured +30-40 % at D256 / D384 L1024 / L768, +3-12 % D512: off)
#endif
// (n2 2026-11-18) inference, D >= 256, item epilogue pipeline: the next item's per-channel vectors and mu_i / rs_i are prefetched (global loads in flight
// during the current item), the output TMA store is issued by warp 3 (no epilogue thread waits for a store); the second and later items of a CTA
// no longer wait ~1.3 us for the previous item's store read + the reload of the residual tile. K3W_X2_<D>: 0 the previous form; 1 also
// two residual / output tiles (one weight-stage ring slot less); 2 one tile.
#ifndef K3W_X2_256
#define K3W_X2_256 1   // D256 mode 1: L128 30.6 -> 30.1, L256 84.5 -> 82.0, L384 176.0 -> 168.8 us (CUPTI span, 3 chains); mode 2 level
#endif
#ifndef K3W_X2_256B
#define K3W_X2_256B 0   // D256 bidirectional (H 512: 12 stages an item): mode 1 measured +2..3 % at L256-L512 (the stage ring is the limit there)
#endif
#ifndef K3W_X2_384
#define K3W_X2_384 0   // (measured: mode 2 level, mode 1 +3 % at L256: one stage ring slot less costs more than the epilogue gains at D384 / D512)
#endif
#ifndef K3W_X2_512
#define K3W_X2_512 0
#endif
#ifndef K3W_SIG
#define K3W_SIG 2   // epilogue sigmoid: 0 ex2 + rcp (two MUFU ops), 2 tanh.approx (one MUFU op; 2026-11-17, k1w / k3g's form)
#endif
#ifndef K3W_EPW
#define K3W_EPW 8
#endif
constexpr int EPW = K3W_EPW, NTHR = 128 + 32 * EPW, ECH = 128 / (EPW / 4);   // epilogue warps, threads, channels per epilogue warp
// OST (inference, 2026-11-18): the LN_out statistics (mu_o / rs_o) computed in this kernel from the t stages it already streams
// (no wide_ln_stats launch): four more warps (warps 4 + EPW ..) read every t stage from shared memory as it lands (thread = 8 tokens
// x 8 channels of the stage, pivot-shifted sums as ln_stats), arrive on the stage's empty barrier with the MMA, and hand the per-token
// mean / rstd of each item to the epilogue through a double-buffered shared-memory table. Items of a token tile already seen by
// this CTA (the grouped schedule) copy the previous item's statistics and leave their stages unread.
constexpr int NSTW = 4, NTHR_OST = NTHR + 32 * NSTW;
constexpr int xm_of(int d, int h) { return K3W_GREG ? 0 : (d == 256 ? (h == 256 ? K3W_X2_256 : K3W_X2_256B) : (d == 384 ? K3W_X2_384 : (d == 512 ? K3W_X2_512 : 0))); }
constexpr int TOK = 128, NTILE = 128, KBB = 16384, STAGE = 2 * KBB;

// the k-th item of CTA c: grouped = token tile c + (k / NTL) G, all NTL channel items of it in a row; else item c + k G
struct Sched {
  int c, G, T, NTL, grouped;
  DEV bool get(int k, int& m0, int& n0) const {
    int tile, half;
    if (grouped) { tile = c + (k / NTL) * G; half = k % NTL; }
    else { const int w = c + k * G; tile = w / NTL; half = w % NTL; }
    m0 = tile * TOK; n0 = half * NTILE;
    return tile < T;
  }
};
// SAVE (training): the epilogue also stores p and g (bf16) for the backward, through two more [128 tok][128 ch] staging tiles;
// the stage ring shrinks from 5 to 3 to make room.
template <int SAVE, int OST = 0, int X2 = 0>
struct Lay {
  static constexpr int NST = SAVE ? 3 : (X2 == 1 ? 4 : 5);   // X2 1: two residual / output tiles instead of one: one stage less
  static constexpr int O_ST = 0, O_XR = O_ST + NST * STAGE, O_PS = O_XR + (X2 == 1 ? 4 : 2) * KBB, O_GS = O_PS + (SAVE ? 2 * KBB : 0),
                       O_VEC = O_GS + (SAVE ? 2 * KBB : 0),
                       O_SP = O_VEC + (X2 ? 2 : 1) * 4 * NTILE * 4,  // OST: [NSTW warps][2 (mean, M2)][TOK] partials
                       O_SB = O_SP + (OST ? NSTW * 2 * TOK * 4 : 0), // OST: [2 items][2 (mu, rs)][TOK]
                       O_BAR = O_SB + (OST ? 2 * 2 * TOK * 4 : 0);
  static constexpr int SMEM = O_BAR + 256 + 1024;
  static_assert(SMEM <= 232448, "smem");
};
constexpr int NSTMAX = 5;
constexpr uint32_t ID_P = idesc_bf16_mj(128, NTILE, 1, 0), ID_G = idesc_bf16_mj(128, NTILE, 0, 0);

struct Bars {
  uint64_t st_full[NSTMAX], st_empty[NSTMAX], acc_full[2], acc_empty[2], xr_full, xr_empty, sf[2], se[2], xr2_full[2], y_ready[2];
  uint32_t tmem;
};

DEV uint32_t sw128(uint32_t r, uint32_t q) { return r * 128u + ((q ^ (r & 7u)) << 4); }
DEV uint4 lds128(uint32_t a) {
  uint4 v;
  asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(a) : "memory");
  return v;
}
DEV float ex2_ftz(float x) { float y; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
DEV float rcp_ftz(float x) { float y; asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
DEV float2 ldf2(const float* p) { return *reinterpret_cast<const float2*>(p); }
DEV uint4 ldg128_k3w(const __nv_bfloat16* p) {
  uint4 v;
  asm volatile("ld.global.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p) : "memory");
  return v;
}
DEV float4 lds128f_k3w(uint32_t a) {
  float4 v;
  asm volatile("ld.shared.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "r"(a));
  return v;
}
// the per-channel epilogue vectors through ld.shared (2026-11-18 pass 12: as a generic float* they compiled to LD.E -- the generic path,
// ~2x the epilogue time); uniform across the warp (lanes = tokens of the same channel): one broadcast wavefront
#ifndef K3W_VLDS
#define K3W_VLDS 1
#endif
DEV float2 lds2f(uint32_t a) { float2 v; asm volatile("ld.shared.v2.f32 {%0,%1}, [%2];" : "=f"(v.x), "=f"(v.y) : "r"(a)); return v; }


// flags (the host's pdl argument): bit 0 launched as a programmatic dependent
template <int D, int H, int SAVE, int OST>
__global__ void __launch_bounds__(OST ? NTHR_OST : NTHR, 1)
    k3w_kernel(const __grid_constant__ CUtensorMap mt, const __grid_constant__ CUtensorMap mwp, const __grid_constant__ CUtensorMap mx,
               const __grid_constant__ CUtensorMap mwg, const __grid_constant__ CUtensorMap my, const __grid_constant__ CUtensorMap mps,
               const __grid_constant__ CUtensorMap mgs, const float* __restrict__ vec,
               const float* __restrict__ mu_o, const float* __restrict__ rs_o, const float* __restrict__ mu_i,
               const float* __restrict__ rs_i, const __nv_bfloat16* __restrict__ ds, int items, int L, int grouped, int flags, float eps,
               const __nv_bfloat16* __restrict__ xg, __nv_bfloat16* __restrict__ yg) {
  constexpr bool GR = K3W_GREG && !SAVE;   // S90: residual from global into registers, y stored from registers (the epilogue off shared memory)
  constexpr int KBT = H / 64, KBX = D / 64, NKB = KBT + KBX, NTL = D / NTILE;
  constexpr int XM = SAVE ? 0 : xm_of(D, H);   // item epilogue pipeline mode (0 off, 1 two residual tiles, 2 one)
  constexpr bool X2 = XM != 0;
  constexpr int XB = XM == 1 ? 2 : 1;       // residual / output tiles
  using LY = Lay<SAVE, OST, XM>;
  constexpr int NST = LY::NST, O_ST = LY::O_ST, O_XR = LY::O_XR, O_PS = LY::O_PS, O_GS = LY::O_GS, O_VEC = LY::O_VEC, O_BAR = LY::O_BAR;
  const int pdl = flags & 1;
  // flags bit 4 (XF, OST inference, pass 15): the x / Wg' stages of a CTA's first item are loaded and multiplied BEFORE griddepcontrol.wait,
  // then the t stages (the previous kernel triggers only after its own wait -- ctr trig 2 -- or completes: x is older than it)
  const int xf = (OST && pdl && (flags & 16)) ? 1 : 0;
  auto kb_of = [&](int j) { return xf ? (j < KBX ? KBT + j : j - KBX) : j; };   // stage j of an item -> K-block (t blocks 0..KBT-1)
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  uint8_t* sm = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t(1023));
  const uint32_t su = smem_u32(sm);
  if (threadIdx.x == 0) K3TL(0)
  Bars& B = *reinterpret_cast<Bars*>(sm + O_BAR);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int cta = blockIdx.x, G = gridDim.x;
  const Sched S{cta, G, items / NTL, NTL, grouped};

  if (tid == 0) {
    for (int s = 0; s < NST; ++s) { mbar_init(&B.st_full[s], 1); mbar_init(&B.st_empty[s], OST ? 1 + NSTW : 1); }
    if (OST) for (int b = 0; b < 2; ++b) { mbar_init(&B.sf[b], 1); mbar_init(&B.se[b], EPW); }
    for (int b = 0; b < 2; ++b) { mbar_init(&B.acc_full[b], 1); mbar_init(&B.acc_empty[b], EPW); }
    mbar_init(&B.xr_full, 1); mbar_init(&B.xr_empty, 1);
    if (X2) for (int b = 0; b < XB; ++b) { mbar_init(&B.xr2_full[b], 1); mbar_init(&B.y_ready[b], 1); }
    fence_mbar_init();
    prefetch_tmap(&mt); prefetch_tmap(&mwp); prefetch_tmap(&mx); prefetch_tmap(&mwg); prefetch_tmap(&my);
  }
  if (warp == 2) { tmem_alloc(&B.tmem, 512); tmem_relinquish(); }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  if (tid == 0) K3TL(1)
  const uint32_t tmem = B.tmem;

  if (warp == 0) {
    if (lane == 0) {
      // ---------------------------------------------------------------- stages: (t block, Wp' block) x KBT, (x block, Wg' block) x KBX
      // OST: the previous kernel is the contraction itself (no ln_stats in between): t is read only after it completed
      if (OST && pdl && !xf) pdl_wait();
      int s = 0; uint32_t ph = 0;
      int m0, n0;
      for (int k = 0; S.get(k, m0, n0); ++k) {
        for (int j = 0; j < NKB; ++j) {
          const int kb = kb_of(j);
          if (xf && k == 0 && j == KBX) pdl_wait();
          mbar_wait(&B.st_empty[s], ph ^ 1);
          if (j == 0) K3T2(k, 0)
          mbar_expect_tx(&B.st_full[s], STAGE);
          uint8_t* a = sm + O_ST + s * STAGE;
          if (kb < KBT) {
            tma_load_2d(a, &mt, &B.st_full[s], m0, kb * 64, EVICT_NORMAL);             // [64 ch][64 tok], tokens 0-63
            tma_load_2d(a + KBB / 2, &mt, &B.st_full[s], m0 + 64, kb * 64, EVICT_NORMAL);   // tokens 64-127
            tma_load_2d(a + KBB, &mwp, &B.st_full[s], kb * 64, n0, EVICT_LAST);
          } else {
            const int kx = kb - KBT;
            tma_load_2d(a, &mx, &B.st_full[s], kx * 64, m0, EVICT_NORMAL);
            tma_load_2d(a + KBB, &mwg, &B.st_full[s], kx * 64, n0, EVICT_LAST);
          }
          if (++s == NST) { s = 0; ph ^= 1; }
        }
      }
    }
  } else if (warp == 3 && !GR) {
    if (lane == 0) {
      // ---------------------------------------------------------------- residual x tile [128 tok][128 ch] (2 K-blocks)
      if (OST && pdl && !xf) pdl_wait();
      int m0, n0;
      if constexpr (X2) {
        // XB residual / output tiles: this thread loads item i's x tile into tile i % XB once the store of item i - XB (same tile) has read it,
        // and issues every item's output store when the epilogue marks the tile written (y_ready): the epilogue warps never wait for a store
        int cnt = 0;
        for (int i = 0; S.get(i, m0, n0); ++i, ++cnt) {
          const int b = XB == 2 ? (i & 1) : 0;
          if (i >= XB) {
            int pm, pn;
            S.get(i - XB, pm, pn);
            mbar_wait(&B.y_ready[b], ((i - XB) / XB) & 1);
            tma_store_2d(&my, sm + O_XR + b * 2 * KBB, pn, pm);
            tma_store_2d(&my, sm + O_XR + b * 2 * KBB + KBB, pn + 64, pm);
            bulk_commit();
            bulk_wait_read<0>();
          }
          mbar_expect_tx(&B.xr2_full[b], 2 * KBB);
          tma_load_2d(sm + O_XR + b * 2 * KBB, &mx, &B.xr2_full[b], n0, m0, EVICT_FIRST);
          tma_load_2d(sm + O_XR + b * 2 * KBB + KBB, &mx, &B.xr2_full[b], n0 + 64, m0, EVICT_FIRST);
        }
        for (int k = cnt > XB ? cnt - XB : 0; k < cnt; ++k) {
          int pm, pn;
          S.get(k, pm, pn);
          const int b = XB == 2 ? (k & 1) : 0;
          mbar_wait(&B.y_ready[b], (k / XB) & 1);
          tma_store_2d(&my, sm + O_XR + b * 2 * KBB, pn, pm);
          tma_store_2d(&my, sm + O_XR + b * 2 * KBB + KBB, pn + 64, pm);
          bulk_commit();
        }
        bulk_wait<0>();
      } else
      for (int i = 0; S.get(i, m0, n0); ++i) {
        if (i >= 1) mbar_wait(&B.xr_empty, (i - 1) & 1);
        mbar_expect_tx(&B.xr_full, 2 * KBB);
        tma_load_2d(sm + O_XR, &mx, &B.xr_full, n0, m0, EVICT_FIRST);
        tma_load_2d(sm + O_XR + KBB, &mx, &B.xr_full, n0 + 64, m0, EVICT_FIRST);
      }
    }
  } else if (warp == 1) {
    // ------------------------------------------------------------------ MMA
    const bool ldr = elect_one();
    int s = 0, m0, n0; uint32_t ph = 0;
    for (int i = 0; S.get(i, m0, n0); ++i) {
      const int b = i & 1;
      mbar_wait(&B.acc_empty[b], ((i >> 1) & 1) ^ 1);
      tc_fence_after();
      const uint32_t dp = tmem + b * 256, dg = dp + 128;
#pragma unroll 1
      for (int j = 0; j < NKB; ++j) {
        const int kb = kb_of(j);
        mbar_wait(&B.st_full[s], ph);
        if (ldr && i < 2 && (kb == 0 || kb == NKB - 1)) K3TL(2 + 3 * i + (kb ? 1 : 0))
        if (ldr && j == 0) K3T2(i, 1)
        if (ldr && j == NKB - 1) K3T2(i, 2)
        tc_fence_after();
        if (ldr) {
          const uint32_t a = su + O_ST + s * STAGE, bw = a + KBB;
          if (kb < KBT) {
#pragma unroll
            for (int k = 0; k < 4; ++k)
              umma_ss(dp, desc_mn_sw128(a + k * 2048, KBB / 2), desc_k_sw128(bw + k * 32), ID_P, (kb | k) != 0);
          } else {
            const int kx = kb - KBT;
#pragma unroll
            for (int k = 0; k < 4; ++k)
              umma_ss(dg, desc_k_sw128(a + k * 32), desc_k_sw128(bw + k * 32), ID_G, (kx | k) != 0);
          }
          umma_commit(&B.st_empty[s]);
        }
        __syncwarp();
        if (++s == NST) { s = 0; ph ^= 1; }
      }
      if (ldr) { umma_commit(&B.acc_full[b]); K3T2(i, 3) }
      __syncwarp();
    }
  } else if (OST && warp >= 4 + EPW) {
    // ------------------------------------------------------------------ LN_out statistics from the t stages (OST)
    // thread = 8 tokens (one 16-byte chunk of a stage row) x 8 channels of every t stage: tc = token chunk 0..15 (half tc / 8),
    // cg = channel group 0..7 (lanes 0-15 / 16-31 of a warp: two groups). Pivot-shifted sums per token as ln_stats_kernel, merged
    // with Chan's formula (lane pairs by shuffle, then the four warps in order through shared memory): deterministic.
    const int st = tid - NTHR, sw = st >> 5, tc = st & 15, cg = st >> 4, hf = tc >> 3, jc = tc & 7;
    float* part = reinterpret_cast<float*>(sm + LY::O_SP);   // [NSTW][2][TOK]
    float* sbuf = reinterpret_cast<float*>(sm + LY::O_SB);   // [2][2][TOK]
    int s = 0, m0, n0, prev = -1; uint32_t ph = 0;
    for (int i = 0; S.get(i, m0, n0); ++i) {
      const int b = i & 1, fresh = (m0 != prev);
      prev = m0;
      float piv[8], sx[8], qx[8];
#pragma unroll
      for (int u = 0; u < 8; ++u) { piv[u] = 0.f; sx[u] = 0.f; qx[u] = 0.f; }
#pragma unroll 1
      for (int j = 0; j < NKB; ++j) {
        const int kb = kb_of(j);
        mbar_wait(&B.st_full[s], ph);
        if (fresh && kb < KBT) {
          const uint32_t base = su + O_ST + s * STAGE + hf * (KBB / 2) + cg * 8 * 128;
          uint4 v[8];
#pragma unroll
          for (int cc = 0; cc < 8; ++cc) v[cc] = lds128(base + cc * 128 + ((jc ^ cc) << 4));
#pragma unroll
          for (int cc = 0; cc < 8; ++cc) {
            const uint32_t w[4] = {v[cc].x, v[cc].y, v[cc].z, v[cc].w};
#pragma unroll
            for (int u2 = 0; u2 < 4; ++u2) {
              const float lo = bf16lo(w[u2]), hi = bf16hi(w[u2]);
              if (kb == 0 && cc == 0) { piv[2 * u2] = lo; piv[2 * u2 + 1] = hi; }
              else {
                const float a = lo - piv[2 * u2], c = hi - piv[2 * u2 + 1];
                sx[2 * u2] += a; qx[2 * u2] = fmaf(a, a, qx[2 * u2]);
                sx[2 * u2 + 1] += c; qx[2 * u2 + 1] = fmaf(c, c, qx[2 * u2 + 1]);
              }
            }
          }
          fence_async_smem();   // generic-proxy reads of the stage before the TMA that refills it (WAR across proxies)
        }
        __syncwarp();
        if (lane == 0) mbar_arrive(&B.st_empty[s]);
        if (++s == NST) { s = 0; ph ^= 1; }
      }
      if (i >= 2) mbar_wait(&B.se[b], ((i >> 1) & 1) ^ 1);   // the epilogue of item i - 2 has read buffer b
      float* dst = sbuf + b * 2 * TOK;
      if (fresh) {
        constexpr float NC = 8.f * KBT;                  // channels per thread
        float mu[8], m2[8];
#pragma unroll
        for (int u = 0; u < 8; ++u) { mu[u] = piv[u] + sx[u] * (1.f / NC); m2[u] = fmaxf(qx[u] - sx[u] * sx[u] * (1.f / NC), 0.f); }
#pragma unroll
        for (int u = 0; u < 8; ++u) {                    // merge with lane ^ 16 (channel group cg ^ 1): equal counts, lower group first
          const float mo_ = __shfl_xor_sync(0xffffffffu, mu[u], 16), qo_ = __shfl_xor_sync(0xffffffffu, m2[u], 16);
          const float ma = (cg & 1) ? mo_ : mu[u], mb = (cg & 1) ? mu[u] : mo_, qa = (cg & 1) ? qo_ : m2[u], qb = (cg & 1) ? m2[u] : qo_;
          const float d = mb - ma;
          mu[u] = ma + 0.5f * d;
          m2[u] = qa + qb + d * d * (0.5f * NC);
        }
        if (lane < 16) {
#pragma unroll
          for (int u = 0; u < 8; ++u) { part[(sw * 2) * TOK + tc * 8 + u] = mu[u]; part[(sw * 2 + 1) * TOK + tc * 8 + u] = m2[u]; }
        }
        named_bar_sync(3, 32 * NSTW);
        {
          const int tk = st;                             // 128 threads = 128 tokens
          float N = 0.f, Mu = 0.f, Q = 0.f;
#pragma unroll
          for (int w = 0; w < NSTW; ++w) {
            const float nb = 2.f * NC, mb = part[(w * 2) * TOK + tk], qb = part[(w * 2 + 1) * TOK + tk];
            const float nt = N + nb, d = mb - Mu;
            Mu += d * (nb / nt);
            Q += qb + d * d * (N * nb / nt);
            N = nt;
          }
          dst[tk] = Mu;
          dst[TOK + tk] = rsqrtf(Q * (1.f / H) + eps);
        }
      } else {
        const float* src = sbuf + (b ^ 1) * 2 * TOK;     // the previous item (same token tile) -- written by these threads
        dst[st] = src[st];
        dst[TOK + st] = src[TOK + st];
      }
      named_bar_sync(3, 32 * NSTW);                      // table complete; part free for the next item
      if (st == 0) { mbar_arrive(&B.sf[b]); K3T2(i, 4) }
    }
  } else if (warp >= 4) {
    // ------------------------------------------------------------------ epilogue: thread = token row r
    const int r = (warp & 3) * 32 + lane, et = tid - 128, c0 = ((warp - 4) >> 2) * ECH;   // c0: this warp's first channel
    const uint32_t lane_off = (uint32_t)((warp & 3) * 32) << 16;
    float* vs = reinterpret_cast<float*>(sm + O_VEC);     // sp, ep, sg, eg of this item's 128 channels
    const uint32_t vsu = su + O_VEC;
#if K3W_VLDS
#define VSL(i) lds2f(vsu + 4u * (uint32_t)(i))
#else
#define VSL(i) ldf2(vs + (i))
#endif
    if (pdl) pdl_wait();   // mu_o / rs_o are the previous kernel's (wide_ln_stats) output (OST: mu_i / rs_i, older); t, x, the weights and vec are older
    int m0, n0;
    constexpr int VPT = (4 * NTILE) / (32 * EPW);   // X2: per-channel vector elements a thread prefetches
    static_assert((4 * NTILE) % (32 * EPW) == 0, "vs fill");
    float mi_c = 0.f, ri_c = 0.f;                   // X2: the carried mu_i / rs_i of this item (loaded one item ahead)
    if constexpr (X2) {
      int fm, fn;
      S.get(0, fm, fn);
      for (int k = et; k < 4 * NTILE; k += 32 * EPW) vs[k] = vec[(k / NTILE) * D + fn + (k % NTILE)];
      named_bar_sync(1, 32 * EPW);
    }
    for (int i = 0; S.get(i, m0, n0); ++i) {
      const int b = i & 1, tok = m0 + r;
      float vnext[VPT], minx = 0.f, rinx = 0.f;
      int m1 = 0, n1 = 0;
      bool hn = false;
      if constexpr (X2) {   // the next item's vectors / statistics: global loads in flight during this item's wait and arithmetic
        hn = S.get(i + 1, m1, n1);
        if (hn) {
#pragma unroll
          for (int q = 0; q < VPT; ++q) { const int k = et + q * 32 * EPW; vnext[q] = vec[(k / NTILE) * D + n1 + (k % NTILE)]; }
          minx = mu_i[m1 + r]; rinx = rs_i[m1 + r];
        }
      } else {
        named_bar_sync(1, 32 * EPW);                     // the previous item's epilogue has finished with vs
        for (int k = et; k < 4 * NTILE; k += 32 * EPW) vs[k] = vec[(k / NTILE) * D + n0 + (k % NTILE)];
        named_bar_sync(1, 32 * EPW);
      }
      const uint32_t vsb = vsu + (X2 ? (uint32_t)b * (16u * NTILE) : 0u);   // X2: vector buffer b
      const int bx = XB == 2 ? b : 0;                                          // X2: residual / output tile
      const uint32_t xrb = su + O_XR + (uint32_t)bx * 2u * KBB;
      if (et == 0 && i < 2) K3TL(16 + i)
      float mo, ro;
      if (OST) {
        mbar_wait(&B.sf[b], (i >> 1) & 1);
        const float* sb = reinterpret_cast<const float*>(sm + LY::O_SB) + b * 2 * TOK;
        mo = sb[r]; ro = sb[TOK + r];
        if (SAVE && n0 == 0) { const_cast<float*>(mu_o)[tok] = mo; const_cast<float*>(rs_o)[tok] = ro; }   // S90: training saves the statistics for the backward
        __syncwarp();
        if (lane == 0) mbar_arrive(&B.se[b]);
      } else {
        mo = mu_o[tok]; ro = rs_o[tok];
      }
      float mi, ri;
      if constexpr (X2) { if (i == 0) { mi = mu_i[tok]; ri = rs_i[tok]; } else { mi = mi_c; ri = ri_c; } }
      else { mi = mu_i[tok]; ri = rs_i[tok]; }
      const float2 ro2 = make_float2(ro, ro), nmo2 = make_float2(-mo, -mo), ri2 = make_float2(ri, ri), nmi2 = make_float2(-mi, -mi);
      // ds [B L, D]: pair row tok = (s L + i) L + j reads row s L + j (tokens b-major; s = 0 for one sample)
      const __nv_bfloat16* dsr = ds ? ds + ((size_t)(tok / (L * L)) * L + tok % L) * D + n0 : nullptr;
      mbar_wait(&B.acc_full[b], (i >> 1) & 1);
      if (et == 0 && i < 2) K3TL(8 + 2 * i)
      if (et == 0) K3T2(i, 5)
      tc_fence_after();
      if constexpr (X2) mbar_wait(&B.xr2_full[bx], (i / XB) & 1);
      else if (!GR) mbar_wait(&B.xr_full, i & 1);
      if (et == 0 && i < 2) K3TL(12 + i)
      if (et == 0) K3T2(i, 7)
      // (n2) the row-dropout scale: without it (inference, no dropout) the multiply by the constant 1.0 and the unpacking of its bf16 words are left out (the same bits)
      auto epi_loop = [&](auto nods_t) {
      constexpr bool NODS = decltype(nods_t)::value;
#pragma unroll 1
      for (int cq = c0 / 32; cq < (c0 + ECH) / 32; ++cq) {   // 32 channels per step
        uint4 xr4[4];
        if (GR) {   // S90: the residual row chunks straight from global (L2) into registers, issued before the TMEM wait
#pragma unroll
          for (int q2 = 0; q2 < 4; ++q2) xr4[q2] = ldg128_k3w(xg + (size_t)tok * D + n0 + cq * 32 + q2 * 8);
        }
        uint32_t pv[32], gv[32];
        tmem_ld32(tmem + lane_off + b * 256 + cq * 32, pv);
        tmem_ld32(tmem + lane_off + b * 256 + 128 + cq * 32, gv);
        tmem_wait_ld();
        if (et == 0 && i < 2 && cq == c0 / 32) K3TL(18 + i)
        if (cq == (c0 + ECH) / 32 - 1) { tc_fence_before(); __syncwarp(); if (lane == 0) mbar_arrive(&B.acc_empty[b]); }
#pragma unroll
        for (int q2 = 0; q2 < 4; ++q2) {                // 8 channels = one 16-byte chunk of the row
          const int cl = cq * 32 + q2 * 8;              // channel within the tile
          const uint32_t addr = xrb + (cl >> 6) * KBB + sw128(r, (cl & 63) >> 3);
          const uint4 xv = GR ? xr4[q2] : lds128(addr);
          uint4 dv = make_uint4(0x3f803f80u, 0x3f803f80u, 0x3f803f80u, 0x3f803f80u);
          if constexpr (!NODS) dv = *reinterpret_cast<const uint4*>(dsr + cl);
          const uint32_t xw[4] = {xv.x, xv.y, xv.z, xv.w}, dw[4] = {dv.x, dv.y, dv.z, dv.w};
          uint32_t o[4], po[4], go[4];
          const uint32_t vb8 = vsb + 4u * (uint32_t)cl;
          const float4 sp4[2] = {lds128f_k3w(vb8), lds128f_k3w(vb8 + 16)}, ep4[2] = {lds128f_k3w(vb8 + 4u * NTILE), lds128f_k3w(vb8 + 4u * NTILE + 16)},
                       sg4[2] = {lds128f_k3w(vb8 + 8u * NTILE), lds128f_k3w(vb8 + 8u * NTILE + 16)}, eg4[2] = {lds128f_k3w(vb8 + 12u * NTILE), lds128f_k3w(vb8 + 12u * NTILE + 16)};
#define V8(a, k) (((k) & 1) ? make_float2((a)[(k) >> 1].z, (a)[(k) >> 1].w) : make_float2((a)[(k) >> 1].x, (a)[(k) >> 1].y))
#define VSP(k) V8(sp4, k)
#define VEP(k) V8(ep4, k)
#define VSG(k) V8(sg4, k)
#define VEG(k) V8(eg4, k)
#pragma unroll
          for (int k = 0; k < 4; ++k) {                 // a channel pair per step on the packed f32x2 pipe
            const int c = cl + 2 * k, j = q2 * 8 + 2 * k;
            const float2 p = __ffma2_rn(ro2, __ffma2_rn(nmo2, VSP(k), make_float2(__uint_as_float(pv[j]), __uint_as_float(pv[j + 1]))),
                                        VEP(k));
            const float2 g = __ffma2_rn(ri2, __ffma2_rn(nmi2, VSG(k),
                                                        make_float2(__uint_as_float(gv[j]), __uint_as_float(gv[j + 1]))),
                                        VEG(k));
#if K3W_SIG == 2   // sigmoid(g) = 0.5 tanh(g / 2) + 0.5: one MUFU op per element (k1w / k3g's form)
            const float2 hg = __fmul2_rn(g, make_float2(0.5f, 0.5f));
            float2 th;
            asm("tanh.approx.f32 %0, %1;" : "=f"(th.x) : "f"(hg.x));
            asm("tanh.approx.f32 %0, %1;" : "=f"(th.y) : "f"(hg.y));
            const float2 sg = __ffma2_rn(th, make_float2(0.5f, 0.5f), make_float2(0.5f, 0.5f));
#else
            const float2 eg = __fmul2_rn(g, make_float2(-1.4426950408889634f, -1.4426950408889634f));
            const float2 dn = __fadd2_rn(make_float2(ex2_ftz(eg.x), ex2_ftz(eg.y)), make_float2(1.f, 1.f));
            const float2 sg = make_float2(rcp_ftz(dn.x), rcp_ftz(dn.y));
#endif
            float2 u = __fmul2_rn(sg, p);
            if constexpr (!NODS) u = __fmul2_rn(u, make_float2(bf16lo(dw[k]), bf16hi(dw[k])));
            const uint32_t ub = pack_bf16(u.x, u.y);    // the update rounded to bf16, then the residual add (the reference's two roundings)
            if (!SAVE) {   // y = bf16(x + ub): one packed bf16 add of the two bf16 words (S90; the training save kernel keeps its form)
              asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(o[k]) : "r"(xw[k]), "r"(ub));
            } else
            {
              const float2 yv = __fadd2_rn(make_float2(bf16lo(xw[k]), bf16hi(xw[k])), make_float2(bf16lo(ub), bf16hi(ub)));
              o[k] = pack_bf16(yv.x, yv.y);
            }
            po[k] = pack_bf16(p.x, p.y);
            go[k] = pack_bf16(g.x, g.y);
          }
          if (GR) *reinterpret_cast<uint4*>(yg + (size_t)tok * D + n0 + cl) = make_uint4(o[0], o[1], o[2], o[3]);
          else sts128(addr, o[0], o[1], o[2], o[3]);
          if (et == 0 && i < 2 && cq == c0 / 32 && q2 == 0) K3TL(20 + i)
          if (SAVE) {
            const uint32_t off = addr - xrb;
            sts128(su + O_PS + off, po[0], po[1], po[2], po[3]);
            sts128(su + O_GS + off, go[0], go[1], go[2], go[3]);
          }
        }
      }
      };
      if (dsr == nullptr) epi_loop(std::true_type{}); else epi_loop(std::false_type{});
      if (et == 0 && i < 2) K3TL(i ? 7 : 4)
      if constexpr (X2) {
        if (hn) {   // the next item's vectors into the other buffer (its last readers finished the previous item before this item's end barrier)
#pragma unroll
          for (int q = 0; q < VPT; ++q) vs[((i + 1) & 1) * 4 * NTILE + et + q * 32 * EPW] = vnext[q];
          mi_c = minx; ri_c = rinx;
        }
        fence_async_smem();
        named_bar_sync(2, 32 * EPW);
        if (et == 0) { if (i < 2) K3TL(9 + 2 * i) mbar_arrive(&B.y_ready[bx]); }
      } else
      if (!GR) {
      fence_async_smem();
      named_bar_sync(2, 32 * EPW);
      if (et == 0) {
        if (i < 2) K3TL(9 + 2 * i)
        K3T2(i, 6)
        tma_store_2d(&my, sm + O_XR, n0, m0);
        tma_store_2d(&my, sm + O_XR + KBB, n0 + 64, m0);
        if (SAVE) {
          tma_store_2d(&mps, sm + O_PS, n0, m0);
          tma_store_2d(&mps, sm + O_PS + KBB, n0 + 64, m0);
          tma_store_2d(&mgs, sm + O_GS, n0, m0);
          tma_store_2d(&mgs, sm + O_GS + KBB, n0 + 64, m0);
        }
        bulk_commit();
        bulk_wait_read<0>();
        mbar_arrive(&B.xr_empty);
      }
      }
    }
    if (!X2 && et == 0) bulk_wait<0>();
  }
  tc_fence_before();
  __syncthreads();
  if (tid == 0) { K3TL(14) }
  if (warp == 2) tmem_dealloc(tmem, 512);
}

int num_sms() {
  static int n = 0;
  if (!n) { int d; cudaGetDevice(&d); cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, d); }
  return n;
}

}  // namespace k3w

// x [M, D], t [H, M], wpq = bf16(Wp g_o) [D, H], wgq = bf16(Wg g_i) [D, D] bf16; vec [4, D] fp32 = (sp, ep, sg, eg);
// mu_o / rs_o / mu_i / rs_i [M] fp32; ds [B L, D] bf16 or none (M = B L L, tokens b-major); y [M, D] bf16; p_out / g_out [M, D] bf16 (training, optional):
// the output projection p and the gate logit g, before the gate.
void k3w_forward(torch::Tensor x, torch::Tensor t, torch::Tensor wpq, torch::Tensor wgq, torch::Tensor vec, torch::Tensor mu_o,
                 torch::Tensor rs_o, torch::Tensor mu_i, torch::Tensor rs_i, c10::optional<torch::Tensor> ds, torch::Tensor y,
                 int64_t L, int64_t grid, c10::optional<torch::Tensor> p_out, c10::optional<torch::Tensor> g_out, int64_t pdl) {
  using namespace k3w;
  const int64_t M = x.size(0), D = x.size(1), H = t.size(0);
  TORCH_CHECK(M % TOK == 0 && t.numel() == H * M && wpq.size(0) == D && wpq.size(1) == H && wgq.size(0) == D && wgq.size(1) == D);
  TORCH_CHECK(x.is_contiguous() && t.is_contiguous() && wpq.is_contiguous() && wgq.is_contiguous() && vec.is_contiguous() &&
              y.is_contiguous());
  TORCH_CHECK(M % (L * L) == 0 && (!ds.has_value() || ds->numel() == M / L * D), "k3w_forward: ds must be [B L, D]");
  const auto bf = CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
  // pdl bits: 0 programmatic dependent; 1 LN_out statistics in the kernel (OST, inference: mu_o / rs_o unused)
  const bool ost = (pdl >> 1) & 1;
  // (S90: with p_out the statistics are also written to mu_o / rs_o by the item whose channel tile starts at 0)
  auto mt = tmap::make(t.data_ptr(), bf, {(uint64_t)M, (uint64_t)H}, {(uint64_t)M * 2}, {64, 64}, CU_TENSOR_MAP_SWIZZLE_128B);
  auto mwp = tmap::make(wpq.data_ptr(), bf, {(uint64_t)H, (uint64_t)D}, {(uint64_t)H * 2}, {64, 128}, CU_TENSOR_MAP_SWIZZLE_128B);
  auto mx = tmap::make(x.data_ptr(), bf, {(uint64_t)D, (uint64_t)M}, {(uint64_t)D * 2}, {64, 128}, CU_TENSOR_MAP_SWIZZLE_128B);
  auto mwg = tmap::make(wgq.data_ptr(), bf, {(uint64_t)D, (uint64_t)D}, {(uint64_t)D * 2}, {64, 128}, CU_TENSOR_MAP_SWIZZLE_128B);
  auto my = tmap::make(y.data_ptr(), bf, {(uint64_t)D, (uint64_t)M}, {(uint64_t)D * 2}, {64, 128}, CU_TENSOR_MAP_SWIZZLE_128B);
  const bool save = p_out.has_value();
  TORCH_CHECK(save == g_out.has_value());
  auto mps = save ? tmap::make(p_out->data_ptr(), bf, {(uint64_t)D, (uint64_t)M}, {(uint64_t)D * 2}, {64, 128}, CU_TENSOR_MAP_SWIZZLE_128B) : my;
  auto mgs = save ? tmap::make(g_out->data_ptr(), bf, {(uint64_t)D, (uint64_t)M}, {(uint64_t)D * 2}, {64, 128}, CU_TENSOR_MAP_SWIZZLE_128B) : my;
  const int items = (int)(M / TOK * (D / NTILE)), tiles = (int)(M / TOK);
  // pdl bit 3 (inference, with OST; 2026-11-18 pass 13): the tile-grouped schedule at every length -- a CTA runs all D / 128 channel
  // items of a token tile back to back, so the LN_out statistics of a tile are computed once (OST) and its t / x K-blocks are L2 hits
  // for the later items -- on min(tiles, SMs) CTAs (L128: one tile a CTA). env TRIMUL_K3W_GRID / TRIMUL_K3W_GROUPED override (A/B).
  const char* gr_env = std::getenv("TRIMUL_K3W_GRID");
  const char* gp_env = std::getenv("TRIMUL_K3W_GROUPED");
  const bool grp_req = gp_env != nullptr ? gp_env[0] == '1' : ((pdl >> 3) & 1) != 0;
  if (grid <= 0 && gr_env != nullptr) grid = std::atoi(gr_env);
  int g = grid > 0 ? (int)grid : (grp_req ? std::min(num_sms(), tiles) : num_sms());
  g = std::min(g, items);
  const int grouped = grp_req || tiles >= 4LL * g;   // enough token tiles: a CTA takes all channel items of a tile in a row (Sched)
  auto st = at::cuda::getCurrentCUDAStream();
  const __nv_bfloat16* dsp = ds.has_value() ? reinterpret_cast<const __nv_bfloat16*>(ds->data_ptr()) : nullptr;
  auto launch = [&](auto kern, int smem, int nthr) {
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    tmap::launch_pdl(kern, dim3(g), dim3(nthr), smem, st, (pdl & 1) != 0, mt, mwp, mx, mwg, my, mps, mgs, (const float*)vec.data_ptr<float>(),
                     (const float*)mu_o.data_ptr<float>(), (const float*)rs_o.data_ptr<float>(), (const float*)mu_i.data_ptr<float>(),
                     (const float*)rs_i.data_ptr<float>(), dsp, items, (int)L, grouped, (int)(pdl & 1) | (int)(pdl & 16), 1e-5f,
                     (const __nv_bfloat16*)x.data_ptr(), (__nv_bfloat16*)y.data_ptr());   // LN eps 1e-5 (the route's contract)
  };
#define K3W_CASE(DD, HH) if (D == DD && H == HH) { \
    constexpr int X2_ = xm_of(DD, HH); \
    if (save && ost) launch(k3w_kernel<DD, HH, 1, 1>, Lay<1, 1>::SMEM, NTHR_OST); \
    else if (save) launch(k3w_kernel<DD, HH, 1, 0>, Lay<1>::SMEM, NTHR); \
    else if (ost) launch(k3w_kernel<DD, HH, 0, 1>, Lay<0, 1, X2_>::SMEM, NTHR_OST); \
    else launch(k3w_kernel<DD, HH, 0, 0>, Lay<0, 0, X2_>::SMEM, NTHR); \
    C10_CUDA_KERNEL_LAUNCH_CHECK(); return; }
  K3W_CASE(256, 512) K3W_CASE(256, 256) K3W_CASE(384, 768) K3W_CASE(384, 384) K3W_CASE(512, 1024) K3W_CASE(512, 512)
  K3W_CASE(128, 128) K3W_CASE(128, 256)   // D128 inference (b200_infer.K3W_D128 cells): one channel item a token tile
#undef K3W_CASE
  TORCH_CHECK(false, "k3w_forward: unsupported (D, H) = (", D, ", ", H, ")");
}

torch::Tensor k3w_tl() {
#ifdef K3W_TL
  torch::Tensor h = torch::empty({160, 32}, torch::kInt64);
  cudaMemcpyFromSymbol(h.data_ptr(), g_k3tl, sizeof(unsigned long long) * 160 * 32);
  return h;
#else
  return torch::empty({0});
#endif
}

torch::Tensor k3w_t2() {
  return torch::empty({0});
}
