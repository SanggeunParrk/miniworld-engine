// k1wt.cu -- (2026-11-18, round 3 training part 8; agent r3-trimul-train) the TRAINING front of the D64 / D128 one-direction path at
// one-ring weight residency (every weight K-block of the CTA's tiles fits the ring: L128): k1w.cu's kernel (front LayerNorm, gated
// projections, token mask -> planes; the same bits) with two changes, so the forms launch leaves the front's critical path:
//   - the four front matrices are read as the stored fp32 master parameters and rounded to bf16 (round to nearest even: the forms
//     launch's rounding, so the operand bits are the same) into the weight ring's SW128 slots by the epilogue warps (idle until the
//     first accumulator), instead of TMA loads of the forms launch's bf16 copies;
//   - it runs BEFORE the forms launch (the reverse of the k1w path): it waits for the previous kernel as k1w does, then triggers, and the
//     forms launch (forms_gs_kernel, a programmatic dependent on the SMs this grid leaves free) does not wait for it at entry -- it reads
//     only the parameters and the RNG scalars, older than this grid's wait -- and waits at its end, so its completion orders this grid
//     before the contraction (an ordinary launch).
// k1w.cu (the inference agent's) is unchanged; this copy is its kernel as of 2026-11-18 with the above edits.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "sm100.cuh"
#include "tmap.h"

using namespace sm100;
#define TL(slot) {}

namespace k1wt {
constexpr int TOK = 128, KBB = TOK * 64 * 2;   // one [128][64] bf16 K-block, SW128 = 16 KB
constexpr int OSTW = 64 * 32 * 2;              // per-warp output box: [64 ch][32 tok] bf16, 64-B swizzle = 4 KB
#ifndef K1W_INSTAT
#define K1W_INSTAT 256   // widths up to this compute the LayerNorm statistics in the transform warps (the row resident in the x ring)
#endif
#ifndef K1W_NWS256
#define K1W_NWS256 6
#endif
#ifndef K1W_EARLY
#define K1W_EARLY 1   // the first ring fills are issued by thread 0 at kernel entry (one-tile latency: L128 / L256)
#endif
#ifndef K1W_PDL
#define K1W_PDL 1   // launched as a programmatic dependent: barrier init, tensor-map prefetch and the TMEM allocation overlap the previous kernel
#endif
#ifndef K1W_XKEEP
#define K1W_XKEEP 24   // x stays in L2 for k3g / the backward (no evict-first hint) while it is at most this many MB
#endif
#ifndef K1W_TRIG
#define K1W_TRIG 1   // launch_dependents at entry: the inference contraction (ctr.cu, a programmatic dependent) schedules its CTAs as k1w's retire
#endif
#ifndef K1W_SPRE
#define K1W_SPRE 1   // D > K1W_INSTAT: k1w_stats (the previous kernel, plain launch) triggers at entry and writes only mean / rstd, so k1w loads
                     // x, the weights and the affine before griddepcontrol.wait; only the transform warps wait, before they read mean / rstd
#endif
#ifndef K1W_NXS256
#define K1W_NXS256 4
#endif
constexpr int NTHREADS = 512;
constexpr float LOG2E = 1.4426950408889634f;
constexpr uint32_t IDESC = idesc_bf16(128, 128);

template <int D>
struct Cfg {
  static constexpr int KB = D / 64;                       // K-blocks per token row
  static constexpr int XC = D / 2;                        // TMEM columns of one xn tile (bf16 pairs)
  static constexpr int NXN = (D <= 256) ? 2 : 1;          // xn tiles resident in TMEM
  static constexpr int NACC = (D <= 128) ? 3 : 2;         // 128-column accumulators
  static constexpr uint32_t T_ACC = 0, T_XN = NACC * 128;
  static_assert(T_XN + NXN * XC <= 512, "tmem");
  static constexpr int NWS = (D <= 128) ? 8 : (D == 256 ? K1W_NWS256 : 6);   // weight K-block ring
  static constexpr int NXS = (D == 256) ? K1W_NXS256 : 4;                    // x K-block ring
  static_assert(D > K1W_INSTAT || D / 64 <= NXS, "in-kernel statistics: the row is resident in the x ring");
  static constexpr int O_W = 0, O_X = O_W + NWS * KBB, O_OUT = O_X + NXS * KBB, O_GB = O_OUT + 8 * OSTW;
  static constexpr int O_BAR = O_GB + 2 * D * 4;
  static constexpr int SMEM = O_BAR + 512 + 1024;
  static_assert(SMEM <= 232448, "smem");
};

struct WSrc { const float* p[4]; long long s0[4]; };   // the fp32 master W_l, W_lg, W_r, W_rg [P / 2, D] (unit column stride)

struct Bars {
  uint64_t w_full[8], w_empty[8], x_full[8], x_empty[8], xn_full[2], xn_empty[2], acc_full[3], acc_empty[3];
  uint32_t tmem;
};

DEV float tanh_approx(float x) { float y; asm("tanh.approx.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
DEV float rsqrt_ftz(float x) { float y; asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
// k1.cu K1_SIG == 2: 0.5 + 0.5 tanh.approx(g / 2)
DEV float2 sigmoid2(float2 g) {
  const float2 h = __fmul2_rn(g, make_float2(0.5f, 0.5f));
  return __ffma2_rn(make_float2(tanh_approx(h.x), tanh_approx(h.y)), make_float2(0.5f, 0.5f), make_float2(0.5f, 0.5f));
}
DEV uint32_t sw128(uint32_t r, uint32_t q) { return r * 128u + ((q ^ (r & 7u)) << 4); }
DEV uint4 lds128(uint32_t a) {
  uint4 v;
  asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(a) : "memory");
  return v;
}
DEV float4 lds128f(uint32_t a) {
  float4 v;
  asm volatile("ld.shared.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "r"(a));
  return v;
}
DEV void tmem_st8(uint32_t taddr, const uint32_t (&r)[8]) {
  asm volatile("tcgen05.st.sync.aligned.32x32b.x8.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8};"
               :: "r"(taddr), "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]), "r"(r[4]), "r"(r[5]), "r"(r[6]), "r"(r[7]) : "memory");
}
DEV void tmem_ld16x256_x4(uint32_t taddr, uint32_t (&r)[16]) {
  asm volatile("tcgen05.ld.sync.aligned.16x256b.x4.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]), "=r"(r[8]),
                 "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15])
               : "r"(taddr) : "memory");
}
DEV void stmatrix_x4_trans(uint32_t addr, uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
  asm volatile("stmatrix.sync.aligned.x4.trans.m8n8.shared.b16 [%0], {%1, %2, %3, %4};" ::"r"(addr), "r"(a), "r"(b), "r"(c), "r"(d)
               : "memory");
}

template <int D, int P, int SMJ>
__global__ void __launch_bounds__(NTHREADS, 1)
    k1wt_kernel(const __grid_constant__ CUtensorMap mx, const __grid_constant__ WSrc ws, const __grid_constant__ CUtensorMap mplane,
               const uint8_t* __restrict__ tokmask, int L, float* __restrict__ mean, float* __restrict__ rstd,
               const float* __restrict__ gamma, const float* __restrict__ beta, int tiles, float eps, int stats_out, int n2, int af) {
  using G_ = Cfg<D>;
  constexpr int KB = G_::KB, XC = G_::XC, NXN = G_::NXN, NACC = G_::NACC, NWS = G_::NWS, NXS = G_::NXS;
  constexpr int NCH = P / 64;
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  uint8_t* sm = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t(1023));
  const uint32_t su = smem_u32(sm);
  Bars& B = *reinterpret_cast<Bars*>(sm + G_::O_BAR);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int cta = blockIdx.x, G = gridDim.x;
  const int n_local = tiles > cta ? (tiles - cta + G - 1) / G : 0;
  const uint64_t xhint = (long long)tiles * TOK * D * 2 <= ((long long)K1W_XKEEP << 20) ? EVICT_NORMAL : EVICT_FIRST;
  // every weight K-block of this CTA's tiles is resident in the ring (the launcher checks n_local NCH KB <= NWS); x as k1w (K1W_EARLY)
  const int nx0 = min(n_local * KB, NXS), nw0 = n_local * NCH * KB;
  if (tid == 0) TL(0)

  if (tid == 0) {
    for (int s = 0; s < NWS; ++s) { mbar_init(&B.w_full[s], 1); mbar_init(&B.w_empty[s], 1); }
    for (int s = 0; s < NXS; ++s) { mbar_init(&B.x_full[s], 1); mbar_init(&B.x_empty[s], 4); }
    for (int b = 0; b < NACC; ++b) { mbar_init(&B.acc_full[b], 1); mbar_init(&B.acc_empty[b], 4); }
    for (int b = 0; b < NXN; ++b) { mbar_init(&B.xn_full[b], 1); mbar_init(&B.xn_empty[b], 1); }
    fence_mbar_init();
    prefetch_tmap(&mx); prefetch_tmap(&mplane);
  }
  if (warp == 2) { tmem_alloc(&B.tmem, 512); tmem_relinquish(); }
  constexpr bool SPRE = K1W_SPRE && D > K1W_INSTAT;   // the wait moves to the transform warps (mean / rstd)
  static_assert(!SPRE, "k1wt: in-kernel statistics only");
  // af (part 12): AFTER the wait-first forms launch, whose output this grid does not read (the masters, x, the mask and the LayerNorm
  // vectors are older than that launch's wait): no wait here, one at the exit (this grid completes after the forms launch)
  if (!af) pdl_wait();   // as k1w: nothing global is read before this
  pdl_trigger();   // after the wait: the forms launch after this one (forms_gs_kernel) does not wait at entry and reads the RNG scalars
  if (tid == 0) {
    // one-tile latency: the first fill of both rings is issued here, before the TMEM allocation and the CTA barrier (a cold box
    // lands ~1.3 us after its issue; the set-up below takes ~1.1 us of that)
    for (int q = 0; q < nx0; ++q) {
      mbar_expect_tx(&B.x_full[q], KBB);
      tma_load_2d(sm + G_::O_X + q * KBB, &mx, &B.x_full[q], (q % KB) * 64, (cta + (q / KB) * G) * TOK, xhint);
    }
  }
  for (int k = tid; k < D; k += NTHREADS) {
    reinterpret_cast<float*>(sm + G_::O_GB)[k] = gamma[k];
    reinterpret_cast<float*>(sm + G_::O_GB)[D + k] = beta[k];
  }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = B.tmem;
  if (tid == 0) TL(1)

  if (warp < 4) {
    if (warp == 0 && lane == 0) {
      // ---------------------------------------------------------------- x K-blocks -> ring (slot free once the transform read it)
      int s = 0, q = 0; uint32_t ph = 0;
      for (int i = 0; i < n_local; ++i) {
        const int row = (cta + i * G) * TOK;
        for (int kb = 0; kb < KB; ++kb) {
          if (q++ >= nx0) {
            mbar_wait(&B.x_empty[s], ph ^ 1);
            mbar_expect_tx(&B.x_full[s], KBB);
            tma_load_2d(sm + G_::O_X + s * KBB, &mx, &B.x_full[s], kb * 64, row, xhint);
          }
          if (++s == NXS) { s = 0; ph ^= 1; }
        }
      }
    } else if (warp == 1) {
      // ---------------------------------------------------------------- MMA: acc[tok][0:64 gate | 64:128 proj] = xn (TMEM) . W_c^T
      const bool ldr = elect_one();
      const uint64_t w0 = desc_k_sw128(su + G_::O_W);
      constexpr uint64_t WSD = KBB >> 4;
      int s = 0, a = 0;
      uint32_t ph = 0, pa = 0;
      for (int i = 0; i < n_local; ++i) {
        const int b = (NXN == 1) ? 0 : (i & 1);
        mbar_wait(&B.xn_full[b], (i / NXN) & 1);
        tc_fence_after();
        const uint32_t xa = tmem + G_::T_XN + b * XC;
#pragma unroll 1
        for (int c = 0; c < NCH; ++c) {
          mbar_wait(&B.acc_empty[a], pa ^ 1);
          tc_fence_after();
          const uint32_t d = tmem + G_::T_ACC + a * 128;
#pragma unroll 1
          for (int kb = 0; kb < KB; ++kb) {
            mbar_wait(&B.w_full[s], ph);
            tc_fence_after();
            if (ldr && i == 0 && c == 0 && kb == 0) TL(4)
            if (ldr) {
              const uint64_t wd = w0 + (uint64_t)s * WSD;
              const uint32_t xk = xa + kb * 32;
              umma_ts(d, xk, wd, IDESC, kb > 0);
              umma_ts(d, xk + 8, wd + 2, IDESC, 1);
              umma_ts(d, xk + 16, wd + 4, IDESC, 1);
              umma_ts(d, xk + 24, wd + 6, IDESC, 1);
              umma_commit(&B.w_empty[s]);
            }
            __syncwarp();
            if (++s == NWS) { s = 0; ph ^= 1; }
          }
          if (ldr) umma_commit(&B.acc_full[a]);
          __syncwarp();
          if (++a == NACC) { a = 0; pa ^= 1; }
        }
        if (ldr && i == 0) TL(5)
        if (ldr) umma_commit(&B.xn_empty[b]);
        __syncwarp();
      }
    }
  } else if (warp < 8) {
    // ------------------------------------------------------------------ transform: thread = token row r, K-block by K-block
    const int r = (warp & 3) * 32 + lane;
    const uint32_t trow = tmem + ((uint32_t)((warp & 3) * 32) << 16);
    auto up = [](uint32_t w) { return make_float2(__uint_as_float(w << 16), __uint_as_float(w & 0xffff0000u)); };
    int s = 0; uint32_t ph = 0;
    for (int i = 0; i < n_local; ++i) {
      const int row = (cta + i * G) * TOK + r, b = (NXN == 1) ? 0 : (i & 1);
      // tokens are b-major: row = (sample * L + i) * L + j, the token mask is [B, L]
      const int ll = L * L, sb = row / ll, rin = row - sb * ll;
      const uint32_t kmask = (tokmask == nullptr || (tokmask[sb * L + rin / L] && tokmask[sb * L + rin % L])) ? 0xffffffffu : 0u;
      float mu, rs;
      if constexpr (D <= K1W_INSTAT) {
        // whole row resident (KB <= 2 ring slots): two-pass statistics here, as k1.cu (packed f32x2 sums)
        int ss = s; uint32_t pp = ph;
        for (int kb = 0; kb < KB; ++kb) { mbar_wait(&B.x_full[ss], pp); if (++ss == NXS) { ss = 0; pp ^= 1; } }
        if (r == 0 && i == 0) TL(2)
        auto ldx = [&](int q) { const int kb = q >> 3; int sl = s + kb; if (sl >= NXS) sl -= NXS;
                                return lds128(su + G_::O_X + sl * KBB + sw128(r, q & 7)); };
        float2 a0 = make_float2(0.f, 0.f), a1 = a0, a2 = a0, a3 = a0;
#pragma unroll (D <= 64 ? D / 8 : 4)   // a full unroll spilled 72 B at D128
        for (int q = 0; q < D / 8; ++q) {
          const uint4 t = ldx(q);
          a0 = __fadd2_rn(a0, up(t.x)); a1 = __fadd2_rn(a1, up(t.y)); a2 = __fadd2_rn(a2, up(t.z)); a3 = __fadd2_rn(a3, up(t.w));
        }
        a0 = __fadd2_rn(__fadd2_rn(a0, a1), __fadd2_rn(a2, a3));
        mu = __fmul_rn(__fadd_rn(a0.x, a0.y), 1.f / D);
        const float2 nm0 = make_float2(-mu, -mu);
        a0 = a1 = a2 = a3 = make_float2(0.f, 0.f);
#pragma unroll (D <= 64 ? D / 8 : 4)
        for (int q = 0; q < D / 8; ++q) {
          const uint4 t = ldx(q);
          float2 d;
          d = __fadd2_rn(up(t.x), nm0); a0 = __ffma2_rn(d, d, a0);
          d = __fadd2_rn(up(t.y), nm0); a1 = __ffma2_rn(d, d, a1);
          d = __fadd2_rn(up(t.z), nm0); a2 = __ffma2_rn(d, d, a2);
          d = __fadd2_rn(up(t.w), nm0); a3 = __ffma2_rn(d, d, a3);
        }
        a0 = __fadd2_rn(__fadd2_rn(a0, a1), __fadd2_rn(a2, a3));
        rs = rsqrt_ftz(__fadd_rn(__fmul_rn(__fadd_rn(a0.x, a0.y), 1.f / D), eps));
        if (stats_out) { mean[row] = mu; rstd[row] = rs; }
      } else {
        if (SPRE && i == 0) pdl_wait();   // mean / rstd are k1w_stats' output
        mu = mean[row]; rs = rstd[row];
      }
      const float2 nm = make_float2(-mu, -mu), rr = make_float2(rs, rs);
      if (i >= NXN) mbar_wait(&B.xn_empty[b], ((i / NXN) - 1) & 1);
      tc_fence_after();
      for (int kb = 0; kb < KB; ++kb) {
        if constexpr (D > K1W_INSTAT) mbar_wait(&B.x_full[s], ph);
        const uint32_t xr = su + G_::O_X + s * KBB;
#pragma unroll
        for (int q4 = 0; q4 < 4; ++q4) {          // 16 channels (8 packed words) per TMEM store
          uint32_t o[8];
#pragma unroll
          for (int qq = 0; qq < 2; ++qq) {
            const int q = q4 * 2 + qq;
            const uint4 t = lds128(xr + sw128(r, q));
            const int col = kb * 64 + q * 8;
            const float4 ga = lds128f(su + G_::O_GB + col * 4);
            const float4 gb4 = lds128f(su + G_::O_GB + col * 4 + 16);
            const float4 ba = lds128f(su + G_::O_GB + D * 4 + col * 4);
            const float4 bb = lds128f(su + G_::O_GB + D * 4 + col * 4 + 16);
            auto aff2 = [&](uint32_t w, float g0, float g1, float b0, float b1) {
              const float2 y = __ffma2_rn(__fmul2_rn(__fadd2_rn(up(w), nm), rr), make_float2(g0, g1), make_float2(b0, b1));
              return pack_bf16(y.x, y.y) & kmask;
            };
            o[4 * qq + 0] = aff2(t.x, ga.x, ga.y, ba.x, ba.y);
            o[4 * qq + 1] = aff2(t.y, ga.z, ga.w, ba.z, ba.w);
            o[4 * qq + 2] = aff2(t.z, gb4.x, gb4.y, bb.x, bb.y);
            o[4 * qq + 3] = aff2(t.w, gb4.z, gb4.w, bb.z, bb.w);
          }
          tmem_st8(trow + G_::T_XN + b * XC + kb * 32 + q4 * 8, o);
        }
        __syncwarp();
        if (lane == 0) mbar_arrive(&B.x_empty[s]);
        if (++s == NXS) { s = 0; ph ^= 1; }
      }
      tmem_wait_st();
      tc_fence_before();
      named_bar_sync(1, 128);
      if (r == 0 && i == 0) TL(3)
      if (warp == 4 && lane == 0) mbar_arrive(&B.xn_full[b]);
    }
  } else {
    // ------------------------------------------------------------------ the weight ring from the fp32 master (warps 8-15, before their
    // epilogue): slot q = (tile, chunk c, K-block kb) holds 64 gate rows (W_lg / W_rg) then 64 projection rows (W_l / W_r) of plane rows
    // 64 c .., columns 64 kb ..; one 16-B group (8 bf16, RNE) per item at sw128(row, group): TMA's SW128 image of the bf16 forms
    {
      const int t = tid - 256;
      for (int q = 0; q < nw0; ++q) {
        const int row = ((q / KB) % NCH) * 64, kb = q % KB, side = row < P / 2 ? 0 : 1, r0 = side ? row - P / 2 : row;
#pragma unroll
        for (int u = 0; u < 4; ++u) {
          const int it = t + 256 * u, rr = it >> 3, g = it & 7, k = rr < 64 ? 2 * side + 1 : 2 * side;
          const float* src = ws.p[k] + (long long)(r0 + (rr & 63)) * ws.s0[k] + kb * 64 + g * 8;
          const float4 a = __ldg(reinterpret_cast<const float4*>(src)), b = __ldg(reinterpret_cast<const float4*>(src + 4));
          uint4 o;
          o.x = pack_bf16(a.x, a.y); o.y = pack_bf16(a.z, a.w); o.z = pack_bf16(b.x, b.y); o.w = pack_bf16(b.z, b.w);
          *reinterpret_cast<uint4*>(sm + G_::O_W + q * KBB + sw128(rr, g)) = o;
        }
      }
      fence_async_smem();   // the generic-proxy stores before the MMA (async proxy) reads them
      named_bar_sync(2, 256);
      if (t == 0) for (int q = 0; q < nw0; ++q) mbar_arrive(&B.w_full[q]);
    }
    // ------------------------------------------------------------------ gate epilogue (k1.cu): warp quarter q = tokens 32q..32q+31
    const int grp = (warp - 8) >> 2, q = warp & 3;
    const uint32_t stg = su + G_::O_OUT + (warp - 8) * OSTW;
    for (int i = 0; i < n_local; ++i) {
      const int row0 = (cta + i * G) * TOK;
      for (int c = grp; c < NCH; c += 2) {
        const int ci = i * NCH + c, acc = ci % NACC;
        mbar_wait(&B.acc_full[acc], (ci / NACC) & 1);
        tc_fence_after();
        if (warp == 8 && lane == 0 && i == 0 && c == 0) TL(6)
        if (warp == 12 && lane == 0 && i == 0 && c == 1) TL(7)
        if (lane == 0) bulk_wait_read<0>();
        __syncwarp();
        const uint32_t tq = tmem + G_::T_ACC + acc * 128 + ((uint32_t)(q * 32) << 16);
        uint32_t ga[16], pa[16], gb[16], pb[16];
        auto ldg = [&](int grp4, uint32_t (&g)[16], uint32_t (&p)[16]) {
          const uint32_t ta = tq + ((uint32_t)((grp4 >> 1) * 16) << 16) + (grp4 & 1) * 32;
          tmem_ld16x256_x4(ta, g);
          tmem_ld16x256_x4(ta + 64, p);
        };
        auto work = [&](int grp4, const uint32_t (&g)[16], const uint32_t (&p)[16]) {
          const int hh = grp4 >> 1, cq = grp4 & 1;
          uint32_t m[8];
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            const float2 s0 = sigmoid2(make_float2(__uint_as_float(g[4 * j]), __uint_as_float(g[4 * j + 1])));
            const float2 s1 = sigmoid2(make_float2(__uint_as_float(g[4 * j + 2]), __uint_as_float(g[4 * j + 3])));
            const float2 a0 = __fmul2_rn(s0, make_float2(__uint_as_float(p[4 * j]), __uint_as_float(p[4 * j + 1])));
            const float2 a1 = __fmul2_rn(s1, make_float2(__uint_as_float(p[4 * j + 2]), __uint_as_float(p[4 * j + 3])));
            m[2 * j] = pack_bf16(a0.x, a0.y);
            m[2 * j + 1] = pack_bf16(a1.x, a1.y);
          }
#pragma unroll
          for (int x = 0; x < 2; ++x) {
            const int k = lane >> 3, jr = lane & 7;
            const int chl = cq * 32 + (2 * x + (k >> 1)) * 8 + jr;
            const int qc = hh * 2 + (k & 1);
            stmatrix_x4_trans(stg + chl * 64 + ((qc ^ ((chl >> 1) & 3)) << 4), m[4 * x], m[4 * x + 1], m[4 * x + 2], m[4 * x + 3]);
          }
        };
        ldg(0, ga, pa);
        tmem_wait_ld();
        ldg(1, gb, pb);
        work(0, ga, pa);
        tmem_wait_ld();
        ldg(2, ga, pa);
        work(1, gb, pb);
        tmem_wait_ld();
        ldg(3, gb, pb);
        work(2, ga, pa);
        tmem_wait_ld();
        tc_fence_before();
        __syncwarp();
        if (lane == 0) mbar_arrive(&B.acc_empty[acc]);
        work(3, gb, pb);
        fence_async_smem();
        __syncwarp();
        if (lane == 0) {
          if constexpr (SMJ != 0) {        // sample-major planes (n2 tokens per sample): the 4-D map of tmap::make_chmajor
            int tq, ts;
            tok_split(row0 + q * 32, n2, tq, ts);
            tma_store_4d(&mplane, sm + G_::O_OUT + (warp - 8) * OSTW, tq, 0, ts, c);
          } else {
            tma_store_2d(&mplane, sm + G_::O_OUT + (warp - 8) * OSTW, row0 + q * 32, c * 64);
          }
          bulk_commit();
        }
      }
    }
    if (warp == 8 && lane == 0) TL(8)
    if (warp == 12 && lane == 0) TL(9)
    if (lane == 0) bulk_wait<0>();
    if (warp == 8 && lane == 0) TL(10)
    if (warp == 12 && lane == 0) TL(11)
  }
  if (af && tid == 0) pdl_wait();
  tc_fence_before();
  __syncthreads();
  if (tid == 0) TL(12)
  if (warp == 2) tmem_dealloc(tmem, 512);
}


template <int D, int P>
void launch(const CUtensorMap& mx, const WSrc& ws, const CUtensorMap& mp, const uint8_t* mask, int L, float* mean, float* rs,
            const float* g, const float* b, int tiles, int grid, cudaStream_t st, float eps, int stats_out, int n2, int af) {
  static bool attr[2] = {false, false};
  auto run = [&](auto kern, bool& at) {
    if (!at) { C10_CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, Cfg<D>::SMEM)); at = true; }
    tmap::launch_pdl(kern, dim3(grid), dim3(NTHREADS), Cfg<D>::SMEM, st, true, mx, ws, mp, mask, L, mean, rs, g, b, tiles, eps, stats_out, n2, af);
  };
  if (n2 > 0) run(k1wt_kernel<D, P, 1>, attr[1]); else run(k1wt_kernel<D, P, 0>, attr[0]);
}
}  // namespace k1wt
#undef TL

// k1w_forward with the four front matrices as the fp32 master parameters [P/2, D] (unit column stride, 16-B aligned rows), D64,
// every weight K-block of a CTA resident (k1wt_fits); a programmatic dependent of the forms launch (see the head comment).
bool k1wt_fits(int64_t D, int64_t P, int64_t M, int64_t grid) {
  using namespace k1wt;
  if (D != 64 || (P != 128 && P != 256) || M % TOK) return false;
  const int64_t tiles = M / TOK, g = std::min<int64_t>(grid, tiles), nl = (tiles + g - 1) / g;
  return nl * (P / 64) <= Cfg<64>::NWS;
}

void k1wt_forward(torch::Tensor x, torch::Tensor wl, torch::Tensor wlg, torch::Tensor wr, torch::Tensor wrg,
                  c10::optional<torch::Tensor> tokmask, torch::Tensor gamma, torch::Tensor beta, torch::Tensor planes, int64_t grid, double eps,
                  bool after_forms) {
  using namespace k1wt;
  const int64_t M = x.size(0), D = x.size(1), P = tmap::chmajor_channels(planes);
  TORCH_CHECK(M % TOK == 0 && planes.numel() == P * M && x.is_contiguous() && planes.is_contiguous() &&
              (planes.dim() == 3 || planes.dim() == 4 || planes.dim() == 5));
  const int L = (int)planes.size(planes.dim() - 1);
  const int64_t nb = M / ((int64_t)L * L);
  TORCH_CHECK(nb * L * L == M && planes.size(planes.dim() - 2) == L);
  TORCH_CHECK(!tokmask.has_value() || (tokmask->numel() == nb * L && tokmask->element_size() == 1 && tokmask->is_contiguous()));
  const int g = (int)std::min<int64_t>(grid, M / TOK);
  TORCH_CHECK(k1wt_fits(D, P, M, g), "k1wt_forward: D64, P 128 / 256, every weight K-block resident");
  const auto bf = CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
  auto mx = tmap::make(x.data_ptr(), bf, {(uint64_t)D, (uint64_t)M}, {(uint64_t)D * 2}, {64, TOK}, CU_TENSOR_MAP_SWIZZLE_128B);
  WSrc ws;
  const torch::Tensor* w4[4] = {&wl, &wlg, &wr, &wrg};
  for (int k = 0; k < 4; ++k) {
    const torch::Tensor& w = *w4[k];
    TORCH_CHECK(w.scalar_type() == torch::kFloat32 && w.size(0) == P / 2 && w.size(1) == D && w.stride(1) == 1 && w.stride(0) % 4 == 0 &&
                    reinterpret_cast<uintptr_t>(w.data_ptr()) % 16 == 0,
                "k1wt_forward: fp32 [P/2, D] master weights, unit column stride, 16-B rows");
    ws.p[k] = w.data_ptr<float>();
    ws.s0[k] = w.stride(0);
  }
  const int n2 = planes.dim() == 5 ? L * L : 0;
  auto mp = n2 ? tmap::make_chmajor(planes, n2, 32u, 64, CU_TENSOR_MAP_SWIZZLE_64B)
               : tmap::make(planes.data_ptr(), bf, {(uint64_t)M, (uint64_t)P}, {(uint64_t)M * 2}, {32u, 64}, CU_TENSOR_MAP_SWIZZLE_64B);
  const uint8_t* mk = tokmask.has_value() ? reinterpret_cast<const uint8_t*>(tokmask->data_ptr()) : nullptr;
  auto st = at::cuda::getCurrentCUDAStream();
  const int af = after_forms ? 1 : 0;
  if (P == 128) launch<64, 128>(mx, ws, mp, mk, L, nullptr, nullptr, gamma.data_ptr<float>(), beta.data_ptr<float>(), (int)(M / TOK), g, st, (float)eps, 0, n2, af);
  else launch<64, 256>(mx, ws, mp, mk, L, nullptr, nullptr, gamma.data_ptr<float>(), beta.data_ptr<float>(), (int)(M / TOK), g, st, (float)eps, 0, n2, af);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("k1wt_fits", &k1wt_fits);
  m.def("k1wt_forward", &k1wt_forward, py::arg("x"), py::arg("wl"), py::arg("wlg"), py::arg("wr"), py::arg("wrg"), py::arg("tokmask"),
        py::arg("gamma"), py::arg("beta"), py::arg("planes"), py::arg("grid"), py::arg("eps"), py::arg("after_forms") = false);
}
