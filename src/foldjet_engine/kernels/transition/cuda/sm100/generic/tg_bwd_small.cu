// tg_bwd_small.cu -- (round 3, training) the generic Transition backward at a few row tiles in ONE launch on the B200 (sm_100a), for the
// single-track cells whose forward is the small-M chain (fused_small_sm100a.chain, which keeps [a | b]): fp32 (TF32) D 384 / 768 and bf16
// D 768, n 2. The generic backward ran 5-6 launches there (tg_split_tf32, GATE_AB, PLAIN split K, tg_ln_bwds, tg_dw), each a few us of
// fixed latency at one to six row tiles; here they are phases of one persistent grid (one CTA per SM, 320 threads) separated by software
// grid barriers:
//
//   P0  (fp32, -DDYRAW=0 only) dy -> dyg = tf32(dy), dylo = tf32(dy - dyg)       every CTA, a grid-stride loop          | barrier
//       (default DYRAW=1: the MMAs read dy as it is, truncated to TF32; no dy_lo term in dWs)
//   P1  GATE_AB tiles: h, [dA | dB] from dy Ws and the forward's [a | b]          tg_gemm.cu MODE 6 (tg_gemm_roles)      | barrier
//   P2  CTAs [0, NPC): the d_xn split-K PLAIN tiles (fp32 slabs)                  tg_gemm.cu MODE 2
//       CTAs [NPC, G): the weight-gradient tiles dWs = dy^T h, dWab = dab^T xn   tg_dw.cu (tg_dw_roles), several per CTA | barrier
//   P3  the LayerNorm backward, one row per warp (row r: CTA r % G, warp r / G): the d_xn slabs added in slab order, dx = dy + ..., and the
//       CTA's dgamma | dbeta partial row (its rows in warp order) into PART[c]                                              | barrier
//   P4  dgb = the PART rows added in CTA order (32-column chunks per CTA, the rows split over the warps, the warp sums in warp order)
//
// The gate and d_xn tiles read the weights as stored (Ws [D][H], [Wa; Wb] [2H][D]) as MN-major B operands (tg_gemm.cu BMN: no transposed
// copies) and store their outputs from registers (GSTORE). Every reduction has one order: a replay is bit-identical. Barriers: monotonic
// counters CNT[0..5] (see grid_sync), CNT[7] the launch number; the buffer is zeroed once per device. All CTAs must be resident: the host
// launches one per SM. Cross-CTA data: each thread orders its writes with fence.proxy.async.global before a barrier and its later TMA
// loads after it; P3 / P4 read with ld.global.cg.
//
// -DF32=<0 | 1> -DDIM=<D> -DHID=<H> -DBNG=<gate tile width> -DBNP=<d_xn tile width> -DNTW=<dW tile width>
// SPDX-License-Identifier: Apache-2.0
#include "sm100.cuh"
#include "../widths/tsx_store.cuh"
using namespace s100;

#ifndef F32
#define F32 0
#endif
#ifndef DIM
#define DIM 384
#endif
#ifndef HID
#define HID 768
#endif
#ifndef BNG
#define BNG 32
#endif
#ifndef BNP
#define BNP 32
#endif
#ifndef DYRAW
#define DYRAW 1                                    // fp32: dy as it is (the MMAs truncate it to TF32; no dy_lo term in dWs): no split phase
#endif
#ifndef GSPLIT
#define GSPLIT 1                                   // P1 as dh = dy Ws split over K (KSG fp32 slabs, all CTAs) + the gate elementwise (P1b)
#endif
#ifndef GST
#define GST 1                                      // the gate / d_xn epilogues store from registers (tg_gemm.cu GSTORE): no bulk-store wait
#endif
#ifndef WFENCE
#define WFENCE 1                                   // the writers' fence.proxy.async.global before a grid barrier (A/B)
#endif
#ifndef PROBE
#define PROBE 0
#endif
#ifndef LNW
#define LNW 1                                      // P3: several warps per row where the CTA's rows leave warps idle (A/B: 0 = one per warp)
#endif
#ifndef LN2SB
#define LN2SB 4                                    // P3 with two warps per row: d_xn slabs per load round
#endif
#ifndef EWU
#define EWU 2                                      // P1b: vectors per thread per load round (A/B: 1 = one per round)
#endif
#ifndef BREL
#define BREL 1                                     // the barrier arrival as atom.add.release.gpu instead of __threadfence + atomicAdd (A/B: 0)
#endif
#ifndef PFSEL
#define PFSEL 1                                    // the proxy fences only at the barriers whose data the next phase TMA-loads (A/B: 0 = all)
#endif
#ifndef NTW
#define NTW 64
#endif

#define TG_GEMM_BODY 1
#define TG_DW_BODY 1
#if PROBE
__device__ unsigned long long* g_stamps;            // (analysis builds) the gate epilogue's trace points: warp 2 lane 0, slots 32 + i
#define TG_TRACE(i) do { if ((threadIdx.x & 31) == 0 && (threadIdx.x >> 5) == 2) { unsigned long long t_; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t_) :: "memory"); g_stamps[blockIdx.x * 40 + 32 + (i)] = t_; } } while (0)
#endif
namespace gate {
#define MODE 6
#define BN BNG
#define BMN 1
#define GSTORE GST
#include "tg_gemm.cu"
#undef MODE
#undef BMN
#undef GSTORE
#undef BN
}  // namespace gate
#undef TG_TRACE
namespace gsk {                                     // (GSPLIT) the gate's products split over K: dh slabs, the gate itself elementwise
#define MODE 2
#define BN BNG
#define BMN 1
#define GSTORE 1
#include "tg_gemm.cu"
#undef MODE
#undef BMN
#undef GSTORE
#undef BN
}  // namespace gsk
namespace plain {
#define MODE 2
#define BN BNP
#define BMN 1
#define GSTORE GST
#include "tg_gemm.cu"
#undef MODE
#undef BMN
#undef GSTORE
#undef BN
}  // namespace plain
namespace dw {
#define NT NTW
#define SPL (F32 && !DYRAW)
#include "tg_dw.cu"
#undef NT
#undef SPL
}  // namespace dw

#if F32
typedef float T;
#else
typedef __nv_bfloat16 T;
#endif

constexpr int NWARP = 10, NTHR = 32 * NWARP;
static_assert(gate::NEPI == 8 && plain::NEPI == 8, "320 threads: the gemm roles' warp layout");
static_assert(DIM % 128 == 0 && DIM <= 1024 && HID % 128 == 0, "widths");
constexpr int NV = DIM / 32;                       // LayerNorm columns per lane (4-wide vectors at 4 (32 i + l))
constexpr int O_BARS = 232448 - 1024;              // the three phases' barriers, past every phase's data
constexpr int RED_BYTES = NWARP * 2 * DIM * 4 + NWARP * 2 * 4;   // P3: the warps' dgamma | dbeta rows, their s1 | s2 partials
constexpr int DATA = gate::O_BAR > plain::O_BAR ? (gate::O_BAR > dw::O_BAR ? gate::O_BAR : dw::O_BAR) : (plain::O_BAR > dw::O_BAR ? plain::O_BAR : dw::O_BAR);
static_assert(DATA <= O_BARS && RED_BYTES <= O_BARS, "shared memory budget");
static_assert(sizeof(gsk::Bars) <= 384 && sizeof(gate::Bars) <= 384 && sizeof(plain::Bars) <= 256 && sizeof(dw::Bars) <= 256, "barrier slots");
constexpr int SMEM_BYTES = O_BARS + 1024;        // (O_BARS + 960: two words; no static shared: it would push the 1 KB-aligned buffer)

DEVI unsigned ld_acquire(const unsigned* p) {
  unsigned v;
  asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
DEVI void fence_proxy_async_global() { asm volatile("fence.proxy.async.global;" ::: "memory"); }
DEVI float4 ldcg4(const float* p) { return __ldcg(reinterpret_cast<const float4*>(p)); }
DEVI float tf32r(float x) { uint32_t r; asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(x)); return __uint_as_float(r); }
DEVI float wsum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}
// 4 activation-dtype values at p as fp32
DEVI void ld4(const T* p, float* v) {
#if F32
  const float4 t = *reinterpret_cast<const float4*>(p);
  v[0] = t.x; v[1] = t.y; v[2] = t.z; v[3] = t.w;
#else
  const uint2 w = *reinterpret_cast<const uint2*>(p);
  v[0] = bf16lo(w.x); v[1] = bf16hi(w.x); v[2] = bf16lo(w.y); v[3] = bf16hi(w.y);
#endif
}
DEVI void st4(T* p, const float* v) {
#if F32
  *reinterpret_cast<float4*>(p) = make_float4(v[0], v[1], v[2], v[3]);
#else
  *reinterpret_cast<uint2*>(p) = make_uint2(pack_bf16(v[0], v[1]), pack_bf16(v[2], v[3]));
#endif
}
// Grid barrier on a monotonic counter: every launch adds G to each of CNT[0..5] (an unused barrier still gets its arrivals), so barrier k
// of the launch numbered GEN = CNT[7] completes at (GEN + 1) G (mod 2^32: the signed difference); the last arrival at barrier 1 bumps GEN
// (every CTA has read it by then). No reset, no exit ticket.
// Every CTA: its writes (generic and async proxy, the TMA stores already complete) before the arrival, the others' after the wait.
// (PROBE: ps, if set, gets 4 stamps: after the proxy fence + CTA barrier, after the gpu fence, after the arrival, after the wait)
DEVI void gstamp(unsigned long long* p) {
#if PROBE
  if (p) { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t) :: "memory"); *p = t; }
#endif
}
// async: the data crossing this barrier is written or read through the async proxy (TMA loads of the next phase), so every thread fences
// its generic writes into it before the arrival and after the wait; generic-only crossings (PFSEL) skip both fences: their order is the
// CTA barrier + thread 0's __threadfence (cumulative) + the acquire load, as cooperative groups' grid sync
DEVI void grid_sync(unsigned* cnt, unsigned target, unsigned* gen = nullptr, unsigned long long* ps = nullptr, bool async = true) {
  __syncwarp();
  if (WFENCE && (async || !PFSEL)) fence_proxy_async_global();
  __syncthreads();
  if (threadIdx.x == 0) {
    gstamp(ps);
#if BREL
    gstamp(ps ? ps + 1 : ps);
    unsigned old;                                  // the arrival as a gpu-scope release (cumulative over the CTA's writes ordered by bar.sync)
    asm volatile("atom.add.release.gpu.global.u32 %0, [%1], 1;" : "=r"(old) : "l"(cnt) : "memory");
#else
    __threadfence();
    gstamp(ps ? ps + 1 : ps);
    const unsigned old = atomicAdd(cnt, 1u);
#endif
    if (gen && old == target - 1u) atomicAdd(gen, 1u);
    gstamp(ps ? ps + 2 : ps);
    while ((int)(ld_acquire(cnt) - target) < 0) {}
    gstamp(ps ? ps + 3 : ps);
  }
  __syncthreads();
  if (async || !PFSEL) fence_proxy_async_global();
}

// P3 of one CTA with WPR warps per row: warp w takes row c + G (w / WPR), columns [(w % WPR) D / WPR, ...); the CTA's dgamma | dbeta
// partial row (its rows in row order) into PART[c]. All threads of the CTA call it (it has CTA barriers). WPR 1 is the one-row-per-warp
// phase as built before (SB0 = 4 slabs per load round, same bits); WPR > 1 loads up to 8 slabs at once (one round for KSP <= 8).
template <int WPR>
DEVI void ln_rows(float* red, int c, int G, int M, int KSP, int MP, int npr, const float* DXN, const T* __restrict__ X,
                  const T* __restrict__ DY, const float* __restrict__ MEAN, const float* __restrict__ RSTD, const float* __restrict__ GAMMA,
                  T* __restrict__ DX, float* PART) {
  constexpr int NVP = NV / WPR, CW = DIM / WPR, NQ = NWARP / WPR;
  static_assert(NVP % 4 == 0 && CW % 128 == 0, "whole 128-column parts");
  constexpr int SB0 = WPR >= 3 ? 8 : WPR == 2 ? LN2SB : 4;     // (WPR 2 at D 768: 12 columns per lane, 8 slabs at once spill)
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int q = warp / WPR, pp = warp % WPR, c0 = pp * CW;
  const int r = c + G * q;                         // row r: CTA r % G, row slot r / G
  const bool has = q < NQ && r < M;
  float* xs = red + NWARP * 2 * DIM;               // (WPR > 1) the warps' s1 | s2 partials
  float dg[NVP], db[NVP], xh[NVP], gx[NVP], ry[NVP], s1 = 0.f, s2 = 0.f, mean = 0.f, rstd = 0.f;
#pragma unroll
  for (int k = 0; k < NVP; ++k) { dg[k] = 0.f; db[k] = 0.f; }
  const long ro = (long)r * DIM;
  if (has) {
    // one round of loads: the row's statistics, its first SB0 d_xn slabs, x, dy, gamma (then any further slabs, SB0 at a time)
    mean = __ldg(MEAN + r); rstd = __ldg(RSTD + r);
    float dd[NVP], xv[NVP], gc[NVP], t[SB0 - 1][NVP];
#pragma unroll
    for (int i = 0; i < NVP / 4; ++i) {
      const int col = c0 + 4 * (32 * i + lane);
      const float4 v = ldcg4(DXN + ro + col);
      dd[4 * i] = v.x; dd[4 * i + 1] = v.y; dd[4 * i + 2] = v.z; dd[4 * i + 3] = v.w;
#pragma unroll
      for (int u = 0; u < SB0 - 1; ++u) {
        if (WPR == 1 || 1 + u < KSP) {
          const int s = 1 + u < KSP ? 1 + u : 0;
          const float4 p = ldcg4(DXN + (long)s * MP * DIM + ro + col);
          t[u][4 * i] = p.x; t[u][4 * i + 1] = p.y; t[u][4 * i + 2] = p.z; t[u][4 * i + 3] = p.w;
        }
      }
      ld4(X + ro + col, xv + 4 * i);
      ld4(DY + ro + col, ry + 4 * i);
      const float4 g = __ldg(reinterpret_cast<const float4*>(GAMMA + col));
      gc[4 * i] = g.x; gc[4 * i + 1] = g.y; gc[4 * i + 2] = g.z; gc[4 * i + 3] = g.w;
    }
#pragma unroll
    for (int u = 0; u < SB0 - 1; ++u)
      if (1 + u < KSP) {
#pragma unroll
        for (int k = 0; k < NVP; ++k) dd[k] += t[u][k];
      }
    for (int s0 = SB0; s0 < KSP; s0 += SB0 - 1) {
#pragma unroll
      for (int u = 0; u < SB0 - 1; ++u) {
        const int s = s0 + u < KSP ? s0 + u : 0;
#pragma unroll
        for (int i = 0; i < NVP / 4; ++i) {
          const float4 p = ldcg4(DXN + (long)s * MP * DIM + ro + c0 + 4 * (32 * i + lane));
          t[u][4 * i] = p.x; t[u][4 * i + 1] = p.y; t[u][4 * i + 2] = p.z; t[u][4 * i + 3] = p.w;
        }
      }
#pragma unroll
      for (int u = 0; u < SB0 - 1; ++u)
        if (s0 + u < KSP) {
#pragma unroll
          for (int k = 0; k < NVP; ++k) dd[k] += t[u][k];
        }
    }
#pragma unroll
    for (int k = 0; k < NVP; ++k) {
      xh[k] = (xv[k] - mean) * rstd;
      gx[k] = dd[k] * gc[k];
      s1 += gx[k];
      s2 += gx[k] * xh[k];
      dg[k] = dd[k] * xh[k];
      db[k] = dd[k];
    }
    s1 = wsum(s1);
    s2 = wsum(s2);
  }
  if (WPR > 1) {                                   // the parts' sums in part order
    if (has && lane == 0) { xs[2 * warp] = s1; xs[2 * warp + 1] = s2; }
    __syncthreads();
    if (has) {
      s1 = xs[2 * q * WPR]; s2 = xs[2 * q * WPR + 1];
#pragma unroll
      for (int j = 1; j < WPR; ++j) { s1 += xs[2 * (q * WPR + j)]; s2 += xs[2 * (q * WPR + j) + 1]; }
    }
  }
  if (has) {
    const float m1 = s1 / (float)DIM, m2 = s2 / (float)DIM;
#pragma unroll
    for (int i = 0; i < NVP / 4; ++i) {
      float o[4];
#pragma unroll
      for (int e = 0; e < 4; ++e) { const int k = 4 * i + e; o[e] = ry[k] + rstd * (gx[k] - m1 - xh[k] * m2); }
      st4(DX + ro + c0 + 4 * (32 * i + lane), o);
    }
  }
  // the CTA's partial row: its rows in row order
  if (c < npr) {
    if (q < NQ) {
#pragma unroll
      for (int i = 0; i < NVP / 4; ++i)
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const int col = c0 + 4 * (32 * i + lane) + e;
          red[q * 2 * DIM + col] = dg[4 * i + e];
          red[q * 2 * DIM + DIM + col] = db[4 * i + e];
        }
    }
    __syncthreads();
    const int nr = (M - c + G - 1) / G;
    const int nw = nr < NQ ? nr : NQ;              // row slots holding a row
    for (int col = tid; col < 2 * DIM; col += NTHR) {
      float a = red[col];
      for (int w = 1; w < nw; ++w) a += red[w * 2 * DIM + col];
      PART[(long)c * 2 * DIM + col] = a;
    }
  }
}

// maps -- gate: gA dyg [M][D] (BKE x 128), gB Ws^T [H][D] (BKE x BNG), gR [a | b] [M][2H] (OCG x 32), gOh h [M][H], gOd [dA | dB] [M][2H]
// (OCG x 32); d_xn: pA [dA | dB] [M][2H] (BKE x 128), pB [Wa; Wb]^T [D][2H] (BKE x BNP), pO slabs [KSP MP][D] fp32 (32 x 32); dW (MN-major,
// EPB x 64): wdy dyg, wh h, wdab [dA | dB], wxn xn, wdylo dylo (fp32).
// DY the incoming gradient (fp32: split in P0 into DYG / DYLO, which the maps above read; bf16: the maps read DY itself), AB the forward's
// [a | b] (gate epilogue), X / MEAN / RSTD / GAMMA the forward's, DX out, DXN the slabs pO writes, DWS [D][H] / DWAB [2H][D] / DGB [2D] fp32
// out, PART [G][2D] fp32 scratch, CNT 8 counters. KSP d_xn K slices (slab stride MP rows; 1: MP = 0), NPC d_xn CTAs.
extern "C" __global__ void __launch_bounds__(NTHR, 1)
tg_bwd_small_sm100(const __grid_constant__ CUtensorMap gA, const __grid_constant__ CUtensorMap gB, const __grid_constant__ CUtensorMap gR,
                   const __grid_constant__ CUtensorMap gOh, const __grid_constant__ CUtensorMap gOd, const __grid_constant__ CUtensorMap pA,
                   const __grid_constant__ CUtensorMap pB, const __grid_constant__ CUtensorMap pO, const __grid_constant__ CUtensorMap wdy,
                   const __grid_constant__ CUtensorMap wh, const __grid_constant__ CUtensorMap wdab, const __grid_constant__ CUtensorMap wxn,
                   const __grid_constant__ CUtensorMap wdylo, const T* __restrict__ DY, float* __restrict__ DYG, float* __restrict__ DYLO,
                   const T* __restrict__ AB, const T* __restrict__ X, const float* __restrict__ MEAN, const float* __restrict__ RSTD,
                   const float* __restrict__ GAMMA, T* __restrict__ DX, float* DXN, float* __restrict__ DWS, float* __restrict__ DWAB,
                   float* __restrict__ DGB, float* PART, unsigned* CNT, T* HB, T* DAB, int M, int KSP, int MP, int NPC, float* GP, int KSG) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  gate::Bars& BG = *reinterpret_cast<gate::Bars*>(sm + O_BARS);
  gsk::Bars& BK = *reinterpret_cast<gsk::Bars*>(sm + O_BARS);   // (GSPLIT: the gate's slot)
  plain::Bars& BP = *reinterpret_cast<plain::Bars*>(sm + O_BARS + 384);
  dw::Bars& BW = *reinterpret_cast<dw::Bars*>(sm + O_BARS + 640);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int c = blockIdx.x, G = gridDim.x;
  const bool dxn_cta = c < NPC;
#if PROBE
  // (analysis builds) per-CTA %globaltimer stamps past the G x 2 D partial rows: [G][40] uint64
  unsigned long long* stamps = reinterpret_cast<unsigned long long*>(PART + (long)G * 2 * DIM);
  if (tid == 0) g_stamps = stamps;
#define ST(i) do { if (tid == 0) { unsigned long long t_; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t_) :: "memory"); stamps[c * 40 + (i)] = t_; } } while (0)
#define STS(i) do { __syncthreads(); ST(i); } while (0)   // after every warp of the CTA
#define PS(i) , stamps + c * 40 + (i)
#define PSA(i) (stamps + c * 40 + (i))
#else
#define ST(i) do { } while (0)
#define STS(i) do { } while (0)
#define PS(i)
#define PSA(i) nullptr
#endif
  if (tid == 0) {
    if (GSPLIT) gsk::tg_gemm_init(BK); else gate::tg_gemm_init(BG);
    if (dxn_cta) plain::tg_gemm_init(BP); else dw::tg_dw_init(BW);
    prefetch_map(&gA); prefetch_map(&gB); prefetch_map(&gR); prefetch_map(&gOh); prefetch_map(&gOd);
  }
  uint32_t& tmem_s = *reinterpret_cast<uint32_t*>(sm + O_BARS + 968);
  if (warp == 1) { tmem_alloc(smem_u32(&tmem_s), 512); tmem_relinquish(); }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = tmem_s;
  ST(0);
  pdl_wait();                                      // dy (and the forward's saves) complete
  unsigned& gen_s = *reinterpret_cast<unsigned*>(sm + O_BARS + 960);
  if (tid == 0) gen_s = ld_acquire(CNT + 7);       // this launch's number (bumped at barrier 1)
  __syncthreads();
  const unsigned target = (gen_s + 1u) * (unsigned)G;

#if F32 && !DYRAW
  // ---------------------------------------------------------------------------------------------------------------- P0: dy split
  {
    const int n4 = M * DIM / 4;
    for (int i = c * NTHR + tid; i < n4; i += G * NTHR) {
      const float4 v = reinterpret_cast<const float4*>(DY)[i];
      const float4 h = make_float4(tf32r(v.x), tf32r(v.y), tf32r(v.z), tf32r(v.w));
      reinterpret_cast<float4*>(DYG)[i] = h;
      reinterpret_cast<float4*>(DYLO)[i] = make_float4(tf32r(v.x - h.x), tf32r(v.y - h.y), tf32r(v.z - h.z), tf32r(v.w - h.w));
    }
    grid_sync(CNT + 0, target);
    ST(1);
  }
#else
  if (tid == 0) atomicAdd(CNT + 0, 1u);           // (barrier 0 unused: keep its count in step with the generation)
#endif

  // -------------------------------------------------------------------------------------------------------------------- P1: gate
#if GSPLIT
  // P1a: dh = dy Ws over KSG K slices (fp32 slabs [KSG][M][H] at GP), every CTA; P1b: the gate elementwise from the slabs and [a | b]
  gsk::tg_gemm_roles(&gA, &gA, &gB, &gB, &gB, &gOh, &gOh, GP, DGB, M, HID, DIM, 0, 0, 0, KSG, M, BK, su, tmem, c, G, GP);
  __syncwarp();                                    // the roles leave lanes 0 (TMA / MMA issue) diverged: reconverge each warp before bar.sync
  tc_fence_before();
  STS(1);
  grid_sync(CNT + 4, target, nullptr, PSA(18), false);   // the dh slabs: generic stores, read with ld.global.cg
  tc_fence_after();
  {
    // EWU vectors per thread per round, every load of a round issued before the first add (the slab loads predicated on KSG: no
    // re-reads of slab 0 for the unused ones); the adds in slab order as before (same bits)
    constexpr int SG = 8;                          // slabs loaded at once
    const int n4 = M * HID / 4;
    const int stride = G * NTHR;
    for (int i0 = c * NTHR + tid; i0 < n4; i0 += EWU * stride) {
      float dh[EWU][4], t[EWU][SG - 1][4], a[EWU][4], b[EWU][4];
#pragma unroll
      for (int v = 0; v < EWU; ++v) {
        const int i = i0 + v * stride;
        if (i < n4) {
          const int r = (4 * i) / HID, j = 4 * i - r * HID;
          const float4 q = ldcg4(GP + 4L * i);
          dh[v][0] = q.x; dh[v][1] = q.y; dh[v][2] = q.z; dh[v][3] = q.w;
#pragma unroll
          for (int u = 0; u < SG - 1; ++u)
            if (1 + u < KSG) {
              const float4 p = ldcg4(GP + (long)(1 + u) * M * HID + 4L * i);
              t[v][u][0] = p.x; t[v][u][1] = p.y; t[v][u][2] = p.z; t[v][u][3] = p.w;
            }
          ld4(AB + (long)r * 2 * HID + j, a[v]);
          ld4(AB + (long)r * 2 * HID + HID + j, b[v]);
        }
      }
#pragma unroll
      for (int v = 0; v < EWU; ++v) {
        const int i = i0 + v * stride;
        if (i >= n4) continue;
        const int r = (4 * i) / HID, j = 4 * i - r * HID;
#pragma unroll
        for (int u = 0; u < SG - 1; ++u)
          if (1 + u < KSG) {
#pragma unroll
            for (int e = 0; e < 4; ++e) dh[v][e] += t[v][u][e];
          }
        for (int s0 = SG; s0 < KSG; ++s0) {
          const float4 q = ldcg4(GP + (long)s0 * M * HID + 4L * i);
          dh[v][0] += q.x; dh[v][1] += q.y; dh[v][2] += q.z; dh[v][3] += q.w;
        }
        float o0[4], o1[4], o2[4];
#pragma unroll
        for (int e = 0; e < 4; ++e) {             // tg_gemm.cu MODE 6's arithmetic
          const float s = sigmoid_kit(a[v][e]), l = a[v][e] * s;
          const float da = dh[v][e] * b[v][e] * (s + l * (1.f - s)), db = dh[v][e] * l;
          o0[e] = F32 ? tf32r(l * b[v][e]) : l * b[v][e];
          o1[e] = F32 ? tf32r(da) : da;
          o2[e] = F32 ? tf32r(db) : db;
        }
        st4(HB + (long)r * HID + j, o0);
        st4(DAB + (long)r * 2 * HID + j, o1);
        st4(DAB + (long)r * 2 * HID + HID + j, o2);
      }
    }
  }
#else
  if (tid == 0) atomicAdd(CNT + 4, 1u);           // (barrier 4 unused: keep its count in step with the generation)
  gate::tg_gemm_roles(&gA, &gA, &gB, &gR, &gB, &gOh, &gOd, AB, DGB, M, HID, DIM, 0, 2 * HID, HID, 1, 0, BG, su, tmem, c, G, HB, DAB);
  __syncwarp();                                    // the roles leave lanes 0 (TMA / MMA issue, store waits) diverged: reconverge each warp
                                                   // before any bar.sync (a partial warp at __syncthreads released it early, measured)
#if PROBE
  if (lane == 0) { unsigned long long t_; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t_) :: "memory"); stamps[c * 40 + 20 + warp] = t_; }
#endif
#endif
  tc_fence_before();
  STS(2);
  grid_sync(CNT + 1, target, CNT + 7 PS(10));
  ST(3);
  tc_fence_after();

  // ---------------------------------------------------------------------------------------------------- P2: d_xn | weight gradients
  if (dxn_cta) {
    if (tid == 0) { prefetch_map(&pA); prefetch_map(&pB); prefetch_map(&pO); }
    plain::tg_gemm_roles(&pA, &pA, &pB, &pB, &pB, &pO, &pO, DXN, DGB, M, DIM, 2 * HID, 0, 0, 0, KSP, MP, BP, su, tmem, c, NPC, DXN);
    __syncwarp();
  } else {
    if (tid == 0) { prefetch_map(&wdy); prefetch_map(&wh); prefetch_map(&wdab); prefetch_map(&wxn); if (F32) prefetch_map(&wdylo); }
    constexpr int DWT = (DIM / 128) * (HID / NTW) + (2 * HID / 128) * (DIM / NTW);
    const int nkb = (M + 63) / 64;
    int tc = 0;
    for (int t = c - NPC; t < DWT; t += G - NPC, ++tc) {
      if (tc > 0) { tc_fence_before(); __syncthreads(); tc_fence_after(); }   // the previous tile's epilogue has read tensor memory
      dw::tg_dw_roles(&wdy, &wh, &wdab, &wxn, &wdylo, DWS, DWS, DWAB, M, DIM, HID, 1, DGB, DGB, 0, BW, su, tmem, t, tc * nkb, tc);
      __syncwarp();                                // reconverge (see P1)
    }
  }
  tc_fence_before();
  STS(4);
  grid_sync(CNT + 2, target, nullptr, PSA(14), !GST);   // d_xn slabs (GSTORE) and dW: generic stores; P3 reads with ld.global.cg
  ST(5);

  // ------------------------------------------------------------------------------------------------- P3: LayerNorm backward rows
  float* red = reinterpret_cast<float*>(sm);
  const int npr = M < G ? M : G;                    // CTAs holding rows = dgamma | dbeta partial rows
  // WPR warps per row where the CTA's rows leave warps idle (LNW): each takes D / WPR columns, all its d_xn slabs in one load round; the
  // row sums meet in shared memory (in part order). Otherwise one row per warp (below; ln_rows<1>, the same arithmetic, measured 2 us
  // slower at L 640 / 768: kept inline)
  const int rpc = (M + G - 1) / G;
  constexpr int D128 = DIM / 128;
  constexpr bool W6 = LNW && D128 % 6 == 0, W3 = LNW && D128 % 3 == 0, W2 = LNW >= 2 && D128 % 2 == 0;   // (two warps per row: slower)
  if (W6 && rpc * 6 <= NWARP) ln_rows<W6 ? 6 : 3>(red, c, G, M, KSP, MP, npr, DXN, X, DY, MEAN, RSTD, GAMMA, DX, PART);
  else if (W3 && rpc * 3 <= NWARP) ln_rows<3>(red, c, G, M, KSP, MP, npr, DXN, X, DY, MEAN, RSTD, GAMMA, DX, PART);
  else if (W2 && rpc * 2 <= NWARP) ln_rows<W2 ? 2 : 3>(red, c, G, M, KSP, MP, npr, DXN, X, DY, MEAN, RSTD, GAMMA, DX, PART);
  else
  {
    const int r = c + G * warp;                   // row r: CTA r % G, warp r / G (one row per warp up to 10 G rows)
    float dg[NV], db[NV];
#pragma unroll
    for (int k = 0; k < NV; ++k) { dg[k] = 0.f; db[k] = 0.f; }
    if (r < M) {
      const long ro = (long)r * DIM;
      // one round of loads: the row's statistics, its first SB0 d_xn slabs, x, dy, gamma (then any further slabs, SB0 at a time)
      constexpr int SB0 = 4;
      const float mean = __ldg(MEAN + r), rstd = __ldg(RSTD + r);
      float dd[NV], xv[NV], ry[NV], gc[NV], t[SB0 - 1][NV];
#pragma unroll
      for (int i = 0; i < NV / 4; ++i) {
        const int col = 4 * (32 * i + lane);
        const float4 q = ldcg4(DXN + ro + col);
        dd[4 * i] = q.x; dd[4 * i + 1] = q.y; dd[4 * i + 2] = q.z; dd[4 * i + 3] = q.w;
#pragma unroll
        for (int u = 0; u < SB0 - 1; ++u) {
          const int s = 1 + u < KSP ? 1 + u : 0;
          const float4 p = ldcg4(DXN + (long)s * MP * DIM + ro + col);
          t[u][4 * i] = p.x; t[u][4 * i + 1] = p.y; t[u][4 * i + 2] = p.z; t[u][4 * i + 3] = p.w;
        }
        ld4(X + ro + col, xv + 4 * i);
        ld4(DY + ro + col, ry + 4 * i);
        const float4 g = __ldg(reinterpret_cast<const float4*>(GAMMA + col));
        gc[4 * i] = g.x; gc[4 * i + 1] = g.y; gc[4 * i + 2] = g.z; gc[4 * i + 3] = g.w;
      }
#pragma unroll
      for (int u = 0; u < SB0 - 1; ++u)
        if (1 + u < KSP) {
#pragma unroll
          for (int k = 0; k < NV; ++k) dd[k] += t[u][k];
        }
      for (int s0 = SB0; s0 < KSP; s0 += SB0 - 1) {
#pragma unroll
        for (int u = 0; u < SB0 - 1; ++u) {
          const int s = s0 + u < KSP ? s0 + u : 0;
#pragma unroll
          for (int i = 0; i < NV / 4; ++i) {
            const float4 p = ldcg4(DXN + (long)s * MP * DIM + ro + 4 * (32 * i + lane));
            t[u][4 * i] = p.x; t[u][4 * i + 1] = p.y; t[u][4 * i + 2] = p.z; t[u][4 * i + 3] = p.w;
          }
        }
#pragma unroll
        for (int u = 0; u < SB0 - 1; ++u)
          if (s0 + u < KSP) {
#pragma unroll
            for (int k = 0; k < NV; ++k) dd[k] += t[u][k];
          }
      }
      float xh[NV], gx[NV], s1 = 0.f, s2 = 0.f;
#pragma unroll
      for (int k = 0; k < NV; ++k) {
        xh[k] = (xv[k] - mean) * rstd;
        gx[k] = dd[k] * gc[k];
        s1 += gx[k];
        s2 += gx[k] * xh[k];
        dg[k] = dd[k] * xh[k];
        db[k] = dd[k];
      }
      const float m1 = wsum(s1) / (float)DIM, m2 = wsum(s2) / (float)DIM;
#pragma unroll
      for (int i = 0; i < NV / 4; ++i) {
        float o[4];
#pragma unroll
        for (int e = 0; e < 4; ++e) { const int k = 4 * i + e; o[e] = ry[k] + rstd * (gx[k] - m1 - xh[k] * m2); }
        st4(DX + ro + 4 * (32 * i + lane), o);
      }
    }
    // the CTA's partial row: its rows (warps) in warp order
    if (c < npr) {
#pragma unroll
      for (int i = 0; i < NV / 4; ++i)
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const int col = 4 * (32 * i + lane) + e;
          red[warp * 2 * DIM + col] = dg[4 * i + e];
          red[warp * 2 * DIM + DIM + col] = db[4 * i + e];
        }
      __syncthreads();
      const int nw = (M - c + G - 1) / G < NWARP ? (M - c + G - 1) / G : NWARP;   // warps holding a row
      for (int col = tid; col < 2 * DIM; col += NTHR) {
        float a = red[col];
        for (int w = 1; w < nw; ++w) a += red[w * 2 * DIM + col];
        PART[(long)c * 2 * DIM + col] = a;
      }
    }
  }
  STS(6);
  // ---------------------------------------------------------------------------------------------------- P4: dgamma | dbeta
  grid_sync(CNT + 3, target, nullptr, nullptr, false);
  ST(7);
  {
    // 32-column chunks over the CTAs; in a chunk warp w adds partial rows [w npr / 10, (w + 1) npr / 10) (at most 15, loaded at once), then
    // warp 0 adds the warp sums in warp order
    float* red4 = reinterpret_cast<float*>(sm);
    constexpr int RW = 15;
    static_assert(RW * NWARP >= 148, "partial rows per warp");
    for (int ch = c; ch < 2 * DIM / 32; ch += G) {
      const int col = ch * 32 + lane;
      const int r0 = warp * npr / NWARP, r1 = (warp + 1) * npr / NWARP;
      float v[RW];
#pragma unroll
      for (int q = 0; q < RW; ++q) v[q] = r0 + q < r1 ? __ldcg(PART + (long)(r0 + q) * 2 * DIM + col) : 0.f;
      float a = 0.f;
#pragma unroll
      for (int q = 0; q < RW; ++q)
        if (r0 + q < r1) a += v[q];
      red4[warp * 32 + lane] = a;
      __syncthreads();
      if (warp == 0) {
        float s = red4[lane];
        for (int w = 1; w < NWARP; ++w) s += red4[w * 32 + lane];
        DGB[col] = s;
      }
      __syncthreads();
    }
  }
  STS(8);
  tc_fence_before();
  __syncthreads();
  if (warp == 1) { tc_fence_after(); tmem_dealloc(tmem, 512); }
}

extern "C" __device__ int transition_smem_bytes = SMEM_BYTES;
