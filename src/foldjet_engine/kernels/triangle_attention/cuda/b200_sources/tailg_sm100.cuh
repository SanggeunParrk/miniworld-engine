// TriangleAttention inference tail with the gate projected from x (the H100 inference tail, tri_tail_sm90<true>), sm_100a:
//   y = LN(x) (bf16, the front's normalisation, two-pass fp32 statistics);  g = y . Wg^T (fp32 in TMEM, never rounded or stored)
//   out = x + (sigmoid(g [+ gb]) o o) . Wo^T [+ bo]
// The front then writes q / k / v only: one [R][HD] bf16 stream less written there and read here.  Used by the fused d_pair 128 path
// (triattn_mod_sm100.cu, CP = HD = 128) and the wide path (triattn_wide_sm100.cu, CP = 64 with HD = 64 / 128).
// CP = d_pair (64 / 128), HD = heads x their padded slot (64 / 128).  Tile = 128 rows.
// The kernel streams: shared memory goes to the x / o rings (the bytes in flight); y and u live in TMEM as the GEMMs' A operands (row m on
// lane m, two bf16 a column); an o slot is free once gated, an x slot once its tile is drained (the residual is read from the slot and
// the result written over it, then stored from there: re-reading the residual rows from global instead cost a third of the kernel).
// Iteration i of warps 2-9 normalises tile i and gates tile i - 1, so the GEMMs of one tile overlap the arithmetic of the next.
// warp 0: TMA x (ring of NXS); warp 14: TMA o (ring of NOS) and Wo | Wg once; warp 1: tcgen05 (G = y . Wg^T, OUT[i % 2] = u . Wo^T);
// warps 2-9: thread = (row = TMEM lane quadrant warp % 4, channel half (warp - 2) / 4): LayerNorm (statistics exchanged with the partner
// warp through shared memory) -> y into Y[i % 2], then u = sigmoid(g) o o -> U[i % 2] (over y when CP = HD); warps 10-13: drain (TMEM
// lane = row) + x -> the result over the x slot's rows -> per-warp TMA stores of [32 rows][64].
#pragma once
#include "sm100.cuh"

#ifndef TG_NOPDL
#define TG_NOPDL 0   // timing only: the tail never waits for the previous grid (WRONG results; the bound of a finer core -> tail hand-off)
#endif
#ifndef TG_HF2
#define TG_HF2 1     // B200 N2 (needs TG_HALF): the gate step multiplies o by 0.5 in packed bf16 (exact) and takes u = h + h tanh(G): one FMUL less a pair (not bitwise the same rounding: fp32 last bit)
#endif
#ifndef TG_HALF
#define TG_HALF 1    // B200 N2: gate GEMM on y / 2 (gamma | beta | gate bias halved at load): sigmoid = 0.5 + 0.5 tanh(G), one FMUL less an element; same bits
#endif
#ifndef TG_STAMP
#define TG_STAMP 0   // timing only: %globaltimer start / end / rank / tiles of every CTA into g_tg_stamp (read by tri_tail_stamps)
#endif
#ifndef TG_NOST
#define TG_NOST 0    // timing only: no output stores
#endif
#ifndef TG_NORES
#define TG_NORES 0   // timing only: no residual loads
#endif
#ifndef TG_NODR
#define TG_NODR 0    // timing only: the drain releases OUT unread
#endif
#ifndef TG_NOLN
#define TG_NOLN 0    // timing only: no LayerNorm arithmetic (y not written)
#endif
#ifndef TG_PD
#define TG_PD 0      // drain software pipeline: the next chunk's TMEM load is issued before this chunk is processed; the slot release of a tile is deferred behind the next tile's first load
#endif
#ifndef TG_DS
#define TG_DS 0      // 1: the starting node (TRANS = 0) stores y straight from registers (no smem write-back / TMA store: the x slot is free once its residual is read)
#endif
#ifndef TG_NXS4
#define TG_NXS4 1    // 1: four x slots at CP = HD = 128 (the shared memory squeezed: no 1 KiB alignment slack, the LN partial sums in one [2][128] buffer)
#endif
#ifndef TG_DW2
#define TG_DW2 0     // 1: a second drain warp group (warps 15-18): group 0 drains K chunk 0 (channels 0-63), group 1 chunk 1 (CP = 128 only)
#endif
#ifndef TG_NOU
#define TG_NOU 0     // timing only: no gate arithmetic (u not written)
#endif
namespace tailg {
using namespace sm100;
__device__ unsigned long long g_tg_stamp[4][512];
__device__ __forceinline__ long long gtimer_tg() { long long t; asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t)); return t; }
#ifndef TG_TL
#define TG_TL 0     // timing only: per-tile stage stamps of CTA 10 into g_tg_tl (tri_tail_tl)
#endif
__device__ unsigned long long g_tg_tl[32][16];
#define TLS(W, EV, IDX) do { if (TG_TL && warp == (W) && lane == 0 && blockIdx.x == 10 && (IDX) < 32) g_tg_tl[(IDX)][(EV)] = (unsigned long long)gtimer_tg(); } while (0)

__device__ __forceinline__ float sig_t(float x) {   // 0.5 + 0.5 tanh(x / 2): one MUFU op
  float t;
  asm("tanh.approx.f32 %0, %1;" : "=f"(t) : "f"(0.5f * x));
  return fmaf(0.5f, t, 0.5f);
}
// B200 N2 (TG_HALF): the gate GEMM runs on y / 2 (the LayerNorm's gamma | beta and the gate bias are halved when loaded: a power-of-two scale is exact in fp32 and bf16,
// so G is exactly g / 2 and sigmoid(g) = 0.5 + 0.5 tanh(G): one FMUL less an element in the gate step, the same bits as sig_t)
__device__ __forceinline__ float sig_g(float x) {
  if (!TG_HALF) return sig_t(x);
  float t;
  asm("tanh.approx.f32 %0, %1;" : "=f"(t) : "f"(x));
  return fmaf(0.5f, t, 0.5f);
}

template <int CP, int HD> struct Cfg {
  static constexpr bool DW2 = TG_DW2 && CP == 128;
  static constexpr int BM = 128, THREADS = 480 + (DW2 ? 128 : 0);
  static constexpr int XT = BM * CP, OT = BM * HD;        // x and o tiles: [K chunks of 64][128][64] 128B-swizzled
  static constexpr bool TIGHT = TG_NXS4 && CP == 128 && HD == 128;
  static constexpr int SRN = TIGHT ? 2 : 4;                // LN partial-sum buffers [SRN][128]
  static constexpr int NXS = CP == 128 ? (TIGHT ? 4 : 3) : 4;            // x slots live from the load to the drain (the result is written over them)
  static constexpr int NOS = CP == 128 ? 1 : HD == 128 ? 3 : 4;   // o slots from the load to the gate step
  // (B200 2026-11-17: NXS / NOS 2 / 2 at CP 128 measured slower: L384 163.8 -> 176.1 us, L768 821 -> 893)
  static constexpr int SMEM = (TIGHT ? 0 : 1024) + (2 * CP * HD + NXS * XT + NOS * OT) * 2 + 2 * CP * 4 + SRN * BM * 4 + 512;
  // TMEM columns: G [HD] fp32 | OUT [2][CP] fp32 | Y [2][CP / 2] bf16 pairs | U [2][HD / 2] (none: over Y when CP = HD)
  static constexpr int TG = 0, TO = HD, TY = HD + 2 * CP, TU = CP == HD ? TY : TY + CP;
  static constexpr int NTM = TU + (CP == HD ? CP : HD);
  static_assert(SMEM <= 232448, "smem");
  static_assert((CP == 64 || CP == 128) && (HD == 64 || HD == 128), "tailg widths");
  static_assert(NTM <= 512, "TMEM");
};

template <int CP, int HD, int OPL = 0>   // OPL 1: o is a channel-major plane [HD][R] (the bias-only chain), loaded as [64 c][128 tokens] boxes
__global__ void __launch_bounds__(Cfg<CP, HD>::THREADS, 1) tri_tailg_sm100(
    int ntiles, int L, float eps,
    const __grid_constant__ CUtensorMap omap,             // o [R][HD] bf16, box (64, 128), 128B swizzle
    const __grid_constant__ CUtensorMap xmap,             // x (the caller's rows) [.][CP], box (64, 128), 128B swizzle
    const __grid_constant__ CUtensorMap wmap,             // Wo [CP][HD] bf16, box (64, CP), 128B swizzle
    const __grid_constant__ CUtensorMap wgmap,            // Wg [HD][CP] bf16, box (64, HD), 128B swizzle
    const __grid_constant__ CUtensorMap ymap,             // out (the caller's rows), box (64, 32), 128B swizzle (TMA stores)
    const __nv_bfloat16* __restrict__ X,                  // x, the caller's [B, Lr, Lr, CP] (the residual)
    const float* __restrict__ LNW, const float* __restrict__ LNB,   // fp32 [CP] each
    const float* __restrict__ GB, const float* __restrict__ OB,     // gate bias fp32 [HD] / out-projection bias fp32 [CP], or nullptr
    const int TRANS,                                      // geometry of x / out (see load_rows)
    __nv_bfloat16* __restrict__ OUTP) {                   // TG_DS: the starting node's output rows (direct stores), else nullptr
  using T = Cfg<CP, HD>;
  constexpr int BM = T::BM, KX = CP / 64, KH = HD / 64, NXS = T::NXS, NOS = T::NOS;
  constexpr int XC = CP / 2, HC = HD / 2;                 // channels of a thread's half (x / y, and g / o / u)
  extern __shared__ __align__(1024) unsigned char raw[];
  unsigned char* smb = raw + ((1024u - (sa(raw) & 1023u)) & 1023u);
  if (T::TIGHT && (sa(raw) & 1023u) != 0u) __trap();   // TIGHT: the dynamic shared memory must start 1 KiB-aligned (no slack left)
  __nv_bfloat16* sW = reinterpret_cast<__nv_bfloat16*>(smb);   // Wo [KH][CP][64]
  __nv_bfloat16* sWg = sW + CP * HD;                             // Wg [KX][HD][64]
  __nv_bfloat16* sX = sWg + CP * HD;                             // [NXS] x
  __nv_bfloat16* sO = sX + NXS * T::XT;                          // [NOS] o
  float* sP = reinterpret_cast<float*>(sO + NOS * T::OT);        // gamma | beta
  float* sR = sP + 2 * CP;                                       // [2 sums][2 halves][128] partial row sums
  uint64_t* bars = reinterpret_cast<uint64_t*>(sR + T::SRN * BM);
  uint64_t* xf = bars;               // [NXS] x landed
  uint64_t* xe = xf + NXS;           // [NXS] count 4: drained and stored (the slot is free)
  uint64_t* of = xe + NXS;           // [NOS] o landed
  uint64_t* oe = of + NOS;           // [NOS] count 8: gated (read)
  uint64_t* yf = oe + NOS;           // [2] count 8: y in TMEM
  uint64_t* gf = yf + 2;             // [2] the gate GEMM retired (G ready, Y[b] read)
  uint64_t* ge = gf + 2;             // count 8: G read
  uint64_t* uf = ge + 1;             // [2] count 8: u in TMEM
  uint64_t* yr = uf + 2;             // [2] the out GEMM retired (U[b] / Y[b] free)
  uint64_t* af = yr + 2;             // [2] OUT ready
  uint64_t* ae = af + 2;             // [2] count 4: OUT drained
  uint64_t* wf = ae + 2;
  uint32_t* tslot = reinterpret_cast<uint32_t*>(wf + 1);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  if (tid == 0) {
    if (TG_STAMP) g_tg_stamp[0][blockIdx.x] = (unsigned long long)gtimer_tg();
    for (int i = 0; i < NXS; ++i) { bar_init(xf + i, 1); bar_init(xe + i, T::DW2 ? 8 : 4); }
    for (int i = 0; i < NOS; ++i) { bar_init(of + i, 1); bar_init(oe + i, 8); }
    for (int i = 0; i < 2; ++i) {
      bar_init(yf + i, 8); bar_init(gf + i, 1); bar_init(uf + i, 8); bar_init(yr + i, 1); bar_init(af + i, 1); bar_init(ae + i, T::DW2 ? 8 : 4);
    }
    bar_init(ge, 8); bar_init(wf, 1);
    bar_init_fence();
  }
  for (int i = tid; i < 2 * CP; i += blockDim.x) sP[i] = (i < CP ? LNW[i] : LNB[i - CP]) * (TG_HALF ? 0.5f : 1.f);
  if (warp == 1) tmem_alloc(tslot, 512);
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = *tslot;
  const int n = ntiles > (int)blockIdx.x ? (ntiles - 1 - (int)blockIdx.x) / (int)gridDim.x + 1 : 0;   // this CTA's tiles
  auto tile = [&](int i) { return (int)blockIdx.x + i * (int)gridDim.x; };
  if (TG_STAMP && tid == 0) { g_tg_stamp[2][blockIdx.x] = (unsigned long long)blockIdx.x; g_tg_stamp[3][blockIdx.x] = (unsigned long long)n; }

  if (warp == 0) {
    if (lane == 0) {
      for (int i = 0; i < n; ++i) {
        const int s = i % NXS;
        if (i >= NXS) wait(xe + s, ((i / NXS) - 1) & 1);
        TLS(0, 0, i);
        expect_tx(xf + s, T::XT * 2);
        for (int kx = 0; kx < KX; ++kx) load_rows(&xmap, sX + s * T::XT + kx * BM * 64, xf + s, kx * 64, tile(i) * BM, L, TRANS);
      }
    }
  } else if (warp == 14) {
    if (lane == 0) {
      expect_tx(wf, 2 * CP * HD * 2);
      for (int kh = 0; kh < KH; ++kh) load_2d(&wmap, sW + kh * CP * 64, wf, kh * 64, 0);
      for (int kx = 0; kx < KX; ++kx) load_2d(&wgmap, sWg + kx * HD * 64, wf, kx * 64, 0);
      if (!TG_NOPDL) pdl_wait();                   // PDL: o is the core's output (x, the weights: written before the front)
      for (int i = 0; i < n; ++i) {
        const int s = i % NOS;
        if (i >= NOS) wait(oe + s, ((i / NOS) - 1) & 1);
        TLS(14, 9, i);
        expect_tx(of + s, T::OT * 2);
        for (int kh = 0; kh < KH; ++kh) {
          if (OPL) load_2d(&omap, sO + s * T::OT + kh * 64 * BM, of + s, tile(i) * BM, kh * 64);   // plane: box (128 tokens, 64 channels)
          else load_2d(&omap, sO + s * T::OT + kh * BM * 64, of + s, kh * 64, tile(i) * BM);
        }
      }
    }
  } else if (warp == 1) {
    if (lane == 0) {
      constexpr uint32_t IDG = idesc_bf16(128, HD, 0, 0), IDO = idesc_bf16(128, CP, 0, 0);
      wait(wf, 0);
      for (int i = 0; i <= n; ++i) {
        if (i < n) {                               // gate GEMM of tile i: G = Y[b] . Wg^T
          const int b = i & 1;
          wait(yf + b, (i >> 1) & 1);
          if (i >= 1) wait(ge, (i - 1) & 1);       // tile i - 1's G has been read
          tc_fence_after();
#pragma unroll
          for (int ks = 0; ks < CP / 16; ++ks)
            mma_ts(tmem + T::TG, tmem + T::TY + b * (CP / 2) + ks * 8, desc_k128(sWg + (ks >> 2) * HD * 64 + (ks & 3) * 16), IDG, ks ? 1u : 0u);
          mma_commit(gf + b);
        }
        if (i >= 1) {                              // out GEMM of tile j = i - 1: OUT[b] = U[b] . Wo^T
          const int j = i - 1, b = j & 1;
          wait(uf + b, (j >> 1) & 1);
          if (j >= 2) wait(ae + b, ((j >> 1) - 1) & 1);
          tc_fence_after();
#pragma unroll
          for (int ks = 0; ks < HD / 16; ++ks)
            mma_ts(tmem + T::TO + b * CP, tmem + T::TU + b * (HD / 2) + ks * 8, desc_k128(sW + (ks >> 2) * CP * 64 + (ks & 3) * 16), IDO, ks ? 1u : 0u);
          mma_commit(af + b);
          mma_commit(yr + b);
        }
      }
    }
  } else if (warp < 10) {
    const int qq = warp & 3, row = qq * 32 + lane, hv = (warp - 2) >> 2;   // TMEM lane quadrant row, channel half
    const uint32_t lq = (uint32_t)(qq * 32) << 16;
    auto soff = [&](int c) { return (c >> 6) * BM * 64 + row * 64 + ((((c & 63) >> 3) ^ (row & 7)) << 3); };   // channel c (a multiple of 8)
    for (int i = 0; i <= n; ++i) {
      if (i < n) {
        // ---- y = LN(x) of tile i (two-pass fp32 statistics, the partner warp's half through shared memory; one bf16 rounding)
        const int s = i % NXS, b = i & 1;
        wait(xf + s, (i / NXS) & 1);
        TLS(2, 1, i);
        const __nv_bfloat16* xt = sX + s * T::XT;
        float2 x[XC / 2];
#pragma unroll
        for (int c8 = 0; c8 < XC / 8; ++c8) {
          const uint4 u = *reinterpret_cast<const uint4*>(xt + soff(hv * XC + c8 * 8));
          x[c8 * 4 + 0] = bf2f(u.x); x[c8 * 4 + 1] = bf2f(u.y); x[c8 * 4 + 2] = bf2f(u.z); x[c8 * 4 + 3] = bf2f(u.w);
        }
        if (TG_NOLN) { if (i >= 2) wait(yr + b, ((i >> 1) - 1) & 1); __syncwarp(); if (lane == 0) arrive(yf + b); goto ustep; }
        float2 s0 = make_float2(0.f, 0.f), s1 = s0;
#pragma unroll
        for (int k = 0; k < XC / 2; k += 2) { s0 = add2(s0, x[k]); s1 = add2(s1, x[k + 1]); }
        sR[hv * BM + row] = s0.x + s0.y + s1.x + s1.y;
        named_sync(1, 256);
        const float mean = (sR[row] + sR[BM + row]) * (1.f / CP);
        if (T::TIGHT) named_sync(1, 256);                   // (one partial-sum buffer: everyone has read the sums before the variances overwrite them)
        const float2 nm = make_float2(-mean, -mean);
        float2 v0 = make_float2(0.f, 0.f), v1 = v0;
#pragma unroll
        for (int k = 0; k < XC / 2; k += 2) {
          x[k] = add2(x[k], nm); x[k + 1] = add2(x[k + 1], nm);
          v0 = fma2(x[k], x[k], v0); v1 = fma2(x[k + 1], x[k + 1], v1);
        }
        sR[(T::SRN - 2 + hv) * BM + row] = v0.x + v0.y + v1.x + v1.y;
        named_sync(1, 256);
        const float rstd = rsqrtf((sR[(T::SRN - 2) * BM + row] + sR[(T::SRN - 1) * BM + row]) * (1.f / CP) + eps);
        if (T::TIGHT) named_sync(1, 256);                   // (and the next tile's sums only after the variances were read)
        const float2 rs = make_float2(rstd, rstd);
        uint32_t yv[XC / 2];
#pragma unroll
        for (int k = 0; k < XC / 2; ++k) {
          const int c = hv * XC + 2 * k;
          const float2 v = fma2(mul2(x[k], rs), make_float2(sP[c], sP[c + 1]), make_float2(sP[CP + c], sP[CP + c + 1]));
          yv[k] = pack2(v.x, v.y);
        }
        if (i >= 2) wait(yr + b, ((i >> 1) - 1) & 1);   // tile i - 2's out GEMM has read Y[b] / U[b]
        tc_fence_after();
#pragma unroll
        for (int k = 0; k < XC / 2; k += 8) tmem_st8(tmem + lq + T::TY + b * (CP / 2) + hv * (XC / 2) + k, yv + k);
        tmem_wait_st();
        tc_fence_before();
        __syncwarp();
        if (lane == 0) arrive(yf + b);
        TLS(2, 2, i);
      }
      ustep:
      if (i >= 1) {
        // ---- u = sigmoid(g [+ gb]) o o of tile j = i - 1 -> U[j % 2] (over its y when CP = HD: the gate GEMM has read it)
        const int j = i - 1, b = j & 1, so = j % NOS;
        TLS(2, 3, j);
        wait(gf + b, (j >> 1) & 1);
        TLS(2, 4, j);
        tc_fence_after();
        float gv[HC / 32][32];
#pragma unroll
        for (int k = 0; k < HC / 32; ++k) tmem_ld32(tmem + lq + T::TG + hv * HC + k * 32, gv[k]);
        tmem_wait_ld();
        tc_fence_before();
        __syncwarp();
        if (lane == 0) arrive(ge);                // G read: the next gate GEMM may overwrite it
        if (TG_NOU) { wait(of + so, (j / NOS) & 1); __syncwarp(); if (lane == 0) arrive(oe + so); if (CP != HD && j >= 2) wait(yr + b, ((j >> 1) - 1) & 1); __syncwarp(); if (lane == 0) arrive(uf + b); continue; }
        if (GB != nullptr) {
#pragma unroll
          for (int k = 0; k < HC; ++k) gv[k >> 5][k & 31] += __ldg(GB + hv * HC + k) * (TG_HALF ? 0.5f : 1.f);
        }
        wait(of + so, (j / NOS) & 1);
        TLS(2, 5, j);
        const __nv_bfloat16* ot = sO + so * T::OT;
        uint32_t uv[HC / 2];
#pragma unroll
        for (int c8 = 0; c8 < HC / 8; ++c8) {
          uint4 ov;
          if (OPL) {   // channel-major plane tile [KH][64 channels][128 tokens]: lanes are consecutive tokens
            const int c = hv * HC + c8 * 8;
            const __nv_bfloat16* pp = ot + (c >> 6) * 64 * BM + (c & 63) * BM + row;
            auto pk = [&](int e) { return (uint32_t)__bfloat16_as_ushort(pp[e * BM]) | ((uint32_t)__bfloat16_as_ushort(pp[(e + 1) * BM]) << 16); };
            ov = make_uint4(pk(0), pk(2), pk(4), pk(6));
          } else {
            ov = *reinterpret_cast<const uint4*>(ot + soff(hv * HC + c8 * 8));
          }
          const uint32_t oo[4] = {ov.x, ov.y, ov.z, ov.w};
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const int k = c8 * 8 + 2 * e;
            if (TG_HF2 && TG_HALF) {   // u = h (1 + tanh G) with h = o / 2 from one packed bf16 multiply (exact): one FMUL less a pair than f * (0.5 + 0.5 tanh G)
              uint32_t hh;
              asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(hh) : "r"(oo[e]), "r"(0x3f003f00u));
              const float2 h = bf2f(hh);
              float t0, t1;
              asm("tanh.approx.f32 %0, %1;" : "=f"(t0) : "f"(gv[k >> 5][k & 31]));
              asm("tanh.approx.f32 %0, %1;" : "=f"(t1) : "f"(gv[(k + 1) >> 5][(k + 1) & 31]));
              uv[c8 * 4 + e] = pack2(fmaf(h.x, t0, h.x), fmaf(h.y, t1, h.y));
            } else {
            const float2 f = bf2f(oo[e]);
            uv[c8 * 4 + e] = pack2(sig_g(gv[k >> 5][k & 31]) * f.x, sig_g(gv[(k + 1) >> 5][(k + 1) & 31]) * f.y);
            }
          }
        }
        __syncwarp();
        if (lane == 0) arrive(oe + so);           // o read (the values are in registers)
        if (CP != HD && j >= 2) wait(yr + b, ((j >> 1) - 1) & 1);   // a separate U[b]: tile j - 2's out GEMM has read it
        tc_fence_after();
#pragma unroll
        for (int k = 0; k < HC / 2; k += 8) tmem_st8(tmem + lq + T::TU + b * (HD / 2) + hv * (HC / 2) + k, uv + k);
        tmem_wait_st();
        tc_fence_before();
        __syncwarp();
        if (lane == 0) arrive(uf + b);
        TLS(2, 6, j);
      }
    }
  } else if (warp < 14 || (T::DW2 && warp >= 15)) {
    // ---- drain of tile j: OUT[j % 2] + the residual from the tile's x slot (TMA-loaded: global loads of the rows cost ~1/3 of the kernel),
    // the result written back over the slot in place and stored by this warp's own TMA stores ([32 rows][64] a store); the slot is
    // released once the stores have read it
    if (!TG_NOPDL) pdl_wait();                     // PDL: the stores come after the core (and everything before it) completed
    const int qq = warp & 3, r = qq * 32 + lane;
    constexpr int NCHD = T::DW2 ? 2 : CP / 32;          // 32-channel chunks a drain warp
    const int chb = (T::DW2 && warp >= 15) ? 2 : 0;     // DW2: the second group drains K chunk 1
    int sp = 0;                                         // (TG_PD) the previous tile's slot, released behind this tile's first TMEM load
    for (int j = 0; j < n; ++j) {
      const int b = j & 1, t = tile(j), s = j % NXS;
      wait(af + b, (j >> 1) & 1);
      TLS(10, 7, j);
      wait(xf + s, (j / NXS) & 1);                // (the slot cannot be refilled before this tile's drain releases it)
      tc_fence_after();
      if (TG_NODR) { tc_fence_before(); __syncwarp(); if (lane == 0) { arrive(ae + b); arrive(xe + s); } continue; }
      __nv_bfloat16* xs = sX + s * T::XT;
      const uint32_t tbase = tmem + ((uint32_t)(qq * 32) << 16) + T::TO + b * CP;
      float vv[2][32];
      if (TG_PD) {                                 // chunk 0's TMEM load, then the previous tile's slot release (its stores had the chain above to finish)
        tmem_ld32(tbase + chb * 32, vv[0]);
        if (TG_PD == 1 && j >= 1 && lane == 0) { bulk_wait_read<0>(); arrive(xe + sp); }
      }
#pragma unroll
      for (int c2 = 0; c2 < NCHD; ++c2) {
        const int ch = chb + c2;
        float (&v)[32] = vv[TG_PD ? (c2 & 1) : 0];
        if (!TG_PD) tmem_ld32(tbase + ch * 32, v);
        // residual: channels 32 ch .. 32 ch + 31 of row r = 16-byte chunks 4 (ch & 1) .. +3 of K chunk ch >> 1; the result goes back there
        __nv_bfloat16* xr = xs + (ch >> 1) * BM * 64 + r * 64;
        uint4 xw4[4];
#pragma unroll
        for (int c = 0; c < 4; ++c)
          xw4[c] = TG_NORES ? make_uint4(0u, 0u, 0u, 0u) : *reinterpret_cast<const uint4*>(xr + ((((ch & 1) * 4 + c) ^ (r & 7)) << 3));
        tmem_wait_ld();
        if (TG_PD && c2 + 1 < NCHD) tmem_ld32(tbase + (ch + 1) * 32, vv[(c2 + 1) & 1]);   // next chunk's load overlaps this chunk's arithmetic and stores
        else if (c2 == NCHD - 1) { tc_fence_before(); __syncwarp(); if (lane == 0) arrive(ae + b); }
#pragma unroll
        for (int c = 0; c < 4; ++c) {
          const uint32_t xw[4] = {xw4[c].x, xw4[c].y, xw4[c].z, xw4[c].w};
          uint32_t w[4];
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const int k = c * 8 + 2 * e;
            float a0 = v[k], a1 = v[k + 1];
            if (OB != nullptr) { a0 += __ldg(OB + ch * 32 + k); a1 += __ldg(OB + ch * 32 + k + 1); }
            const float2 xx = bf2f(xw[e]);
            w[e] = pack2(a0 + xx.x, a1 + xx.y);
          }
          if (TG_DS && OUTP != nullptr) *reinterpret_cast<uint4*>(OUTP + (long)(t * BM + r) * CP + ch * 32 + c * 8) = make_uint4(w[0], w[1], w[2], w[3]);
          else *reinterpret_cast<uint4*>(xr + ((((ch & 1) * 4 + c) ^ (r & 7)) << 3)) = make_uint4(w[0], w[1], w[2], w[3]);
        }
        if ((ch & 1) && !(TG_DS && OUTP != nullptr)) {   // this warp's 32 rows of the K chunk: one TMA store
          fence_proxy_async();
          __syncwarp();
          if (lane == 0) {
            if (!TG_NOST) store_rows(&ymap, xs + (ch >> 1) * BM * 64 + qq * 32 * 64, (ch >> 1) * 64, t * BM + qq * 32, L, TRANS);
            bulk_commit();
          }
        }
      }
      if (TG_PD == 1) sp = s;
      else if (TG_DS && OUTP != nullptr) { __syncwarp(); if (lane == 0) arrive(xe + s); }   // the slot was only read
      else { if (lane == 0) { bulk_wait_read<0>(); arrive(xe + s); } __syncwarp(); }
      TLS(10, 8, j);
    }
    if (lane == 0) { if (TG_PD == 1 && n > 0) { bulk_wait_read<0>(); arrive(xe + sp); } bulk_wait<0>(); }
  }
  tc_fence_before();
  __syncthreads();
  if (TG_STAMP && tid == 0) g_tg_stamp[1][blockIdx.x] = (unsigned long long)gtimer_tg();
  if (warp == 1) tmem_dealloc(tmem, 512);
}

// out = x + (sigmoid(LN(x) . Wg^T [+ gb]) o o) . Wo^T [+ bo]: o [R, HD] bf16 (the kernels' rows, R = B L^2, L a multiple of 128); x the
// caller's [B, Lr, Lr, CP] bf16 (geometry trans / lr as the other row kernels); wo [CP, HD], wg [HD, CP] bf16 contiguous (rows at stride
// CP: a row block of the front's packed weights); lnw / lnb fp32 [CP]; gb fp32 [HD] / ob fp32 [CP] or empty; out_: a contiguous bf16
// tensor with B Lr^2 CP elements to write into, or empty (a new tensor)
template <int CP, int HD, int OPL = 0>
torch::Tensor launch(const torch::Tensor& o, const torch::Tensor& x, const torch::Tensor& wo, const torch::Tensor& wg, const torch::Tensor& lnw,
                     const torch::Tensor& lnb, double eps, int64_t L, const torch::Tensor& out_, bool trans, int64_t lr, const torch::Tensor& gb,
                     const torch::Tensor& ob) {
  using T = Cfg<CP, HD>;
  TORCH_CHECK(o.is_cuda() && o.scalar_type() == torch::kBFloat16 && o.is_contiguous() && o.numel() % ((long)HD * T::BM) == 0, "o: [R, HD] bf16");
  const long R = o.numel() / HD;
  TORCH_CHECK(R % (L * L) == 0 && L % T::BM == 0, "rows = B L^2, L % 128");
  const int geo = rows_geo(trans, L, lr);
  const long Lr = geo_rows_len(L, geo), Rr = R / (L * L) * Lr * Lr;
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kBFloat16 && x.is_contiguous() && x.numel() == Rr * CP, "x: [B, Lr, Lr, CP] bf16");
  TORCH_CHECK(wo.scalar_type() == torch::kBFloat16 && wo.is_contiguous() && wo.numel() == (long)CP * HD, "wo [CP, HD] bf16");
  TORCH_CHECK(wg.scalar_type() == torch::kBFloat16 && wg.is_contiguous() && wg.numel() == (long)CP * HD, "wg [HD, CP] bf16");
  for (const auto* p : {&lnw, &lnb}) TORCH_CHECK(p->scalar_type() == torch::kFloat32 && p->is_contiguous() && p->numel() == CP, "lnw / lnb fp32 [CP]");
  const bool hg = gb.defined() && gb.numel() > 0, ho = ob.defined() && ob.numel() > 0;
  if (hg) TORCH_CHECK(gb.scalar_type() == torch::kFloat32 && gb.is_contiguous() && gb.numel() == HD, "gb fp32 [HD]");
  if (ho) TORCH_CHECK(ob.scalar_type() == torch::kFloat32 && ob.is_contiguous() && ob.numel() == CP, "ob fp32 [CP]");
  const bool given = out_.defined() && out_.numel() > 0;
  if (given) TORCH_CHECK(out_.scalar_type() == torch::kBFloat16 && out_.is_contiguous() && out_.numel() == Rr * CP && out_.device() == o.device(),
                         "out: contiguous bf16 with B Lr^2 x CP elements");
  auto out = given ? out_.view({Rr, CP}) : torch::empty({Rr, CP}, o.options());
  CUtensorMap om = OPL ? make_map<2>(o.data_ptr(), {(uint64_t)R, (uint64_t)HD}, {(uint64_t)R}, {128, 64}, CU_TENSOR_MAP_SWIZZLE_NONE, "o plane")
                       : make_map<2>(o.data_ptr(), {(uint64_t)HD, (uint64_t)R}, {(uint64_t)HD}, {64, 128}, CU_TENSOR_MAP_SWIZZLE_128B, "o");
  CUtensorMap wm = make_map<2>(wo.data_ptr(), {(uint64_t)HD, (uint64_t)CP}, {(uint64_t)HD}, {64, (uint32_t)CP}, CU_TENSOR_MAP_SWIZZLE_128B, "wo");
  CUtensorMap wgm = make_map<2>(wg.data_ptr(), {(uint64_t)CP, (uint64_t)HD}, {(uint64_t)CP}, {64, (uint32_t)HD}, CU_TENSOR_MAP_SWIZZLE_128B, "wg");
  CUtensorMap xm = make_rows_map(x.data_ptr(), CP, R, CP, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B, "x", L, geo);
  CUtensorMap ym = make_rows_map(out.data_ptr(), CP, R, CP, 64, 32, CU_TENSOR_MAP_SWIZZLE_128B, "out", L, geo);
  static bool attr = false;
  if (!attr) { C10_CUDA_CHECK(cudaFuncSetAttribute(tri_tailg_sm100<CP, HD, OPL>, cudaFuncAttributeMaxDynamicSharedMemorySize, T::SMEM)); attr = true; }
  const int ntiles = (int)(R / T::BM);
  launch_pdl(tri_tailg_sm100<CP, HD, OPL>, dim3(std::min(ntiles, num_sms(o.device().index()))), dim3(T::THREADS), T::SMEM, at::cuda::getCurrentCUDAStream(),
      ntiles, (int)L, (float)eps, om, xm, wm, wgm, ym, reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()), lnw.data_ptr<float>(),
      lnb.data_ptr<float>(), hg ? gb.data_ptr<float>() : nullptr, ho ? ob.data_ptr<float>() : nullptr, geo,
      (TG_DS && geo == 0) ? reinterpret_cast<__nv_bfloat16*>(out.data_ptr()) : nullptr);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
}  // namespace tailg
