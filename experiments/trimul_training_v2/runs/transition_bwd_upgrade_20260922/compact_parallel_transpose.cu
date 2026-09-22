// transition_bwd.cu — the Transition backward pass (LayerNorm + SwiGLU expand + squeeze + residual) of the MiniWorld pair
// Transition at D = 128, H = 4D = 512, bf16 activations and weights, as ONE fused sm_90a kernel.
// SPDX-License-Identifier: Apache-2.0
// Built on the Anthropic native v5 device primitives (`vendor/anthropic_v5/csrc/tmn_ptx.cuh` via `tmn_kernels.cuh`: TMA,
// wgmma, ldmatrix / stmatrix, mbarrier, the 128-byte swizzle and the quad reduction).  The schedule follows the two-role split
// of the B1-B4 TriMul training kernel.
//
// Contract (the engine's saved-xn + fused-residual backward, with its rounding points and multiplication order):
//   dh = bf16(dy Ws)      a = xn Wa^T, b = xn Wb^T (fp32)      sig = 1/(1 + 2^(-a log2 e))      silu = a sig
//   h  = bf16(silu b)     dA = bf16((dh b)(sig + silu(1 - sig)))      dB = bf16(dh silu)
//   dWs = dy^T h, dWa = dA^T xn, dWb = dB^T xn (fp32 sums, rounded once)      d_xn = bf16(dA Wa + dB Wb)
//   LayerNorm backward from the saved statistics (rstd, c1 = mean rstd):
//     xhat = (x - mean) rstd, wdy = gamma d_xn, dx = bf16(bf16((wdy - xhat ca - cb) rstd) + dy)
//     dgamma += d_xn xhat, dbeta += d_xn (fp32)
//
// Two CTA roles in one launch, no cluster and no cooperative launch:
//   DW CTAs (8 hidden slices x DW_REPL replicas) own a 64-unit hidden slice and keep its Ws / Wa / Wb resident (48 KB).  Both
//     warpgroups run the gate stage on their own 64 rows of a 128-row tile, then split the weight gradients over the whole tile
//     (WG0: dWa_s and the left half of dWs^T; WG1: dWb_s and the right half), 96 fp32 accumulators per thread.
//   DX CTAs stream all eight 64-unit hidden chunks of their 128-row tile through a two-slot TMA weight ring.  Per chunk each
//     warpgroup computes dh and the packed [a|b], forms dA / dB as bf16 directly in the wgmma A-fragment registers (the m64n64 C
//     fragment IS the m64k64 A fragment, so there is no shared-memory round trip), and accumulates d_xn with an RS m64n128 over
//     the same packed [Wa; Wb] operand.  d_xn never leaves registers; the LayerNorm backward, the residual and the dgamma /
//     dbeta partials follow in the same fragment layout.
//   Only dh, a and b are recomputed (by the DW role), so the kernel executes 22 M D H FLOP against a 16 M D H minimum.  A first
//     design that avoided even that, by splitting the hidden axis over a cluster and reduce-scattering the d_xn partial sums
//     through distributed shared memory, was 2.5x slower: see README.md.
//
// Partials: `partw` is [NDW][3][64 hs][128 d] fp32 (dWa_s, dWb_s, dWs^T_s) and `dgbw` is [NDX][8 warps][2][128] fp32
// (dgamma, dbeta) - one private row per warp, so no atomic is needed anywhere.  `reduce_partials` sums both.
#include "tmn_kernels.cuh"
using namespace tmn; using namespace tmn::sm90;

#ifndef DW_REPL
#define DW_REPL 8
#endif
#ifndef NCTA
#define NCTA 132
#endif
constexpr int D_ = 128, H_ = 512, HS = 64, NCH = H_ / HS, ROWS = 128, WGR = 64;
constexpr int NDW = 8 * DW_REPL, NDX = NCTA - NDW;
static_assert(NDX > 0, "no DX CTAs left");

// ---- DW role shared memory
constexpr int W_WS = 0, W_WAB = 16384;                         // 48 KB resident slice weights: Ws_s 16 KB, then [Wa_s; Wb_s] 32 KB
constexpr int W_IN = 49152, W_INB = 65536;                     // per buffer: dy 32 KB at +0, xn 32 KB at +32768 (2 column blocks of 16 KB)
constexpr int W_XN = 32768;
constexpr int W_HDB = W_IN + 2 * W_INB;                        // h, dA, dB: [128 rows][64 hs] bf16, 16 KB each
constexpr int W_BAR = W_HDB + 3 * 16384;
// ---- DX role shared memory
constexpr int X_RING = 0, X_SLOT = 49152;                      // 2 slots: Ws_j 16 KB at +0, then [Wa_j; Wb_j] 32 KB at +16384
constexpr int X_IN = 2 * X_SLOT, X_INB = 65536, X_XN = 32768;  // per buffer: dy 32 KB, xn 32 KB (the xn half is reloaded with x for the epilogue)
constexpr int X_DGB = X_IN + 2 * X_INB;                        // [2][128] fp32 dgamma / dbeta partials (shared-memory atomics)
constexpr int X_GAM = X_DGB + 1024;                            // gamma, fp32 [128]
constexpr int X_BAR = X_GAM + 512;
constexpr int SMEM_BYTES = 231424;
static_assert(W_BAR + 256 <= SMEM_BYTES && X_BAR + 256 <= SMEM_BYTES, "shared memory budget");

TMN_DEVI float warp_sum(float x) {
#pragma unroll
  for (int k = 16; k; k >>= 1) x += __shfl_xor_sync(0xffffffffu, x, k);
  return x;
}
TMN_DEVI float sigmoid_kit(float a) {                 // math::sigmoid of the Anthropic kit: rcp.approx.ftz(1 + ex2.approx.ftz(-a log2 e))
  return rcpf(__fadd_rn(1.f, ex2f(__fmul_rn(-1.4426950408889634f, a))));   // (math::sigmoid); div.full is a multi-instruction sequence
}
TMN_DEVI uint32_t lds32(uint32_t a) { uint32_t v; asm volatile("ld.shared.b32 %0, [%1];" : "=r"(v) : "r"(a) : "memory"); return v; }
TMN_DEVI void stg32u(void* p, uint32_t v) { asm volatile("st.global.b32 [%0], %1;" :: "l"(p), "r"(v) : "memory"); }
TMN_DEVI uint64_t dsc_(uint32_t addr, uint32_t lbo, uint32_t sbo) { return smem_desc(addr, lbo, sbo, 1); }
TMN_DEVI uint64_t dk64(uint32_t base, int ks) { return dsc_(base + (ks >> 2) * 8192 + (ks & 3) * 32, 16, 1024); }    // K-major over a 64-row tile
TMN_DEVI uint64_t dk128(uint32_t base, int ks) { return dsc_(base + (ks >> 2) * 16384 + (ks & 3) * 32, 16, 1024); }  // K-major over a 128-row tile
TMN_DEVI uint64_t dmn(uint32_t base, int ks, uint32_t lbo) { return dsc_(base + ks * 2048, lbo, 1024); }             // MN-major, k-step = 16 rows

template <int TA, int TB> TMN_DEVI void mma64(float (&d)[32], uint64_t a, uint64_t b, int accumulate) {
  asm volatile("{ .reg .pred p; setp.ne.b32 p, %34, 0; wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, %32, %33, p, 1, 1, %35, %36; }"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]) : "l"(a), "l"(b), "r"(accumulate), "n"(TA), "n"(TB));
}
template <int TA, int TB> TMN_DEVI void mma128(float (&d)[64], uint64_t a, uint64_t b, int accumulate) {
  asm volatile("{ .reg .pred p; setp.ne.b32 p, %66, 0; wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, %64, %65, p, 1, 1, %67, %68; }"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63]) : "l"(a), "l"(b), "r"(accumulate), "n"(TA), "n"(TB));
}
TMN_DEVI void mma128_rs(float (&d)[64], const uint32_t (&a)[4], uint32_t desc_lo, uint32_t desc_hi, uint32_t off16, int accumulate) {
  asm volatile("{\n .reg .pred p;\n .reg .b32 lo;\n .reg .b64 dsc;\n setp.ne.b32 p, %71, 0;\n add.u32 lo, %68, %70;\n mov.b64 dsc, {lo, %69};\n"
    "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, {%64,%65,%66,%67}, dsc, p, 1, 1, 1;\n}\n"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(desc_lo), "r"(desc_hi), "r"(off16), "r"(accumulate));
}

struct BwdPar {                                 // the tensor maps stay in the grid-constant parameter bank; Par only carries their addresses
  const CUtensorMap *dy, *xn, *x, *ws, *wa, *wb;
  const float *rstd, *c1, *gamma;
  __nv_bfloat16* dx;
  float *dgam, *dbeta, *partw;      // partw: [NDW][3][64 hs][128 d] fp32 (0 = dWa_s, 1 = dWb_s, 2 = dWs^T_s)
  float* dgbw;                      // dgbw:  [NDX][8 warps][2][128] fp32
  int M, tiles;
};

// ============================================================================================ DW role
TMN_DEVI void weight_role(const BwdPar& p, uint8_t* sm, int cta, int tid, int wg, int wtid, int warp, int lane) {
  const int slice = cta % 8, repl = cta / 8, nrep = DW_REPL;
  const uint32_t su = smem_u32(sm);
  uint64_t* bars = reinterpret_cast<uint64_t*>(sm + W_BAR);
  uint64_t* w_full = bars; uint64_t* in_full = bars + 1; uint64_t* in_free = bars + 3; uint64_t* hdb_free = bars + 5;
  const int n_local = (p.tiles > repl) ? (p.tiles - repl + nrep - 1) / nrep : 0;
  auto issue_in = [&](int i) {
    const int buf = i & 1, row = (repl + i * nrep) * ROWS; uint8_t* b = sm + W_IN + buf * W_INB;
    mbar_arrive_expect_tx(in_full + buf, 65536);
#pragma unroll
    for (int c = 0; c < 2; ++c)
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        tma_load_2d(b + c * 16384 + h * 8192, p.dy, in_full + buf, c * 64, row + h * 64);
        tma_load_2d(b + W_XN + c * 16384 + h * 8192, p.xn, in_full + buf, c * 64, row + h * 64);
      }
  };
  if (tid == 0) {
    mbar_init(w_full, 1); mbar_init(in_full, 1); mbar_init(in_full + 1, 1); mbar_init(in_free, 2); mbar_init(in_free + 1, 2);
    mbar_init(hdb_free, 2);
    fence_barrier_init();
  }
  __syncthreads();
  if (tid == 0) {
    mbar_arrive_expect_tx(w_full, 49152);
    tma_load_2d(sm + W_WS, p.ws, w_full, slice * HS, 0);
#pragma unroll
    for (int c = 0; c < 2; ++c) {                                     // [Wa_s; Wb_s] as [128 n][128 d]: 2 column blocks of 16 KB
      tma_load_2d(sm + W_WAB + c * 16384, p.wa, w_full, c * 64, slice * HS);
      tma_load_2d(sm + W_WAB + c * 16384 + 8192, p.wb, w_full, c * 64, slice * HS);
    }
    for (int i = 0; i < 2 && i < n_local; ++i) issue_in(i);
  }
  float accW1[64], accW2[32];
#pragma unroll
  for (int e = 0; e < 64; ++e) accW1[e] = 0.f;
#pragma unroll
  for (int e = 0; e < 32; ++e) accW2[e] = 0.f;
  mbar_wait(w_full, 0);
  for (int i = 0; i < n_local; ++i) {
    const int buf = i & 1;
    const uint32_t inu = su + W_IN + buf * W_INB, hdbu = su + W_HDB;
    mbar_wait(in_full + buf, (i >> 1) & 1);
    // ---- stage 1: this warpgroup's 64 rows -> dh, a, b -> gate -> h, dA, dB in the shared tile
    float acc[32], AB[64];
#pragma unroll
    for (int e = 0; e < 32; ++e) acc[e] = 0.f;
#pragma unroll
    for (int e = 0; e < 64; ++e) AB[e] = 0.f;
    fence_regs(acc); fence_regs(AB); wgmma_fence();
#pragma unroll
    for (int ks = 0; ks < 8; ++ks) mma64<0, 1>(acc, dk128(inu + wg * 8192, ks), dmn(su + W_WS, ks, 16), ks > 0);
    wgmma_commit();
#pragma unroll
    for (int ks = 0; ks < 8; ++ks) mma128<0, 0>(AB, dk128(inu + W_XN + wg * 8192, ks), dk128(su + W_WAB, ks), ks > 0);
    wgmma_commit();
    wgmma_wait<1>(); fence_regs(acc);
    uint32_t dhp[16];
#pragma unroll
    for (int e = 0; e < 16; ++e) dhp[e] = pack_bf16(acc[2 * e], acc[2 * e + 1]);
    wgmma_wait<0>(); fence_regs(AB);
    if (i >= 1) mbar_wait(hdb_free, (i - 1) & 1);          // both warpgroups retired the previous tile's weight-gradient GEMMs
    {
      const int m = lane >> 3, row = wg * WGR + 16 * warp + 8 * (m & 1) + (lane & 7);
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        uint32_t hp[4], dap[4], dbp[4];
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          const int e = 4 * j + q;
          const float a0 = AB[4 * (e >> 1) + 2 * (e & 1)], a1 = AB[4 * (e >> 1) + 2 * (e & 1) + 1];
          const float b0 = AB[4 * ((e >> 1) + 8) + 2 * (e & 1)], b1 = AB[4 * ((e >> 1) + 8) + 2 * (e & 1) + 1];
          const float g0 = bf16lo(dhp[e]), g1 = bf16hi(dhp[e]);
          const float s0 = sigmoid_kit(a0), s1 = sigmoid_kit(a1), l0 = a0 * s0, l1 = a1 * s1;
          hp[q] = pack_bf16(l0 * b0, l1 * b1);
          dap[q] = pack_bf16((g0 * b0) * (s0 + l0 * (1.f - s0)), (g1 * b1) * (s1 + l1 * (1.f - s1)));
          dbp[q] = pack_bf16(g0 * l0, g1 * l1);
        }
        const uint32_t off = swz128((uint32_t)row, (uint32_t)((2 * j + (m >> 1)) * 16));
        stsm_x4(hdbu + off, hp[0], hp[1], hp[2], hp[3]);
        stsm_x4(hdbu + 16384 + off, dap[0], dap[1], dap[2], dap[3]);
        stsm_x4(hdbu + 32768 + off, dbp[0], dbp[1], dbp[2], dbp[3]);
      }
    }
    fence_proxy_async();
    named_bar_sync(1, 256);                                 // both warpgroups' halves of h / dA / dB are in place
    // ---- stage 2: the weight gradients of the whole 128-row tile, split between the warpgroups
    fence_regs(accW1); fence_regs(accW2); wgmma_fence();
#pragma unroll
    for (int ks = 0; ks < 8; ++ks) mma128<1, 1>(accW1, dmn(hdbu + (wg == 0 ? 16384 : 32768), ks, 16), dmn(inu + W_XN, ks, 16384), 1);
    wgmma_commit();
#pragma unroll
    for (int ks = 0; ks < 8; ++ks) mma64<1, 1>(accW2, dmn(hdbu, ks, 16), dmn(inu + wg * 16384, ks, 16), 1);
    wgmma_commit();
    wgmma_wait<0>(); fence_regs(accW1); fence_regs(accW2);
    if (wtid == 0) { mbar_arrive(hdb_free); mbar_arrive(in_free + buf); }
    if (tid == 0 && i + 2 < n_local) { mbar_wait(in_free + buf, (i >> 1) & 1); issue_in(i + 2); }
  }
  // ---- fp32 partials of this CTA
  float* base = p.partw + (size_t)cta * 3 * HS * D_;
  const int r0 = 16 * warp + (lane >> 2);
#pragma unroll
  for (int g = 0; g < 16; ++g) {
    const int c = 8 * g + 2 * (lane & 3);
    stg64f(base + (size_t)wg * HS * D_ + r0 * D_ + c, accW1[4 * g], accW1[4 * g + 1]);
    stg64f(base + (size_t)wg * HS * D_ + (r0 + 8) * D_ + c, accW1[4 * g + 2], accW1[4 * g + 3]);
  }
#pragma unroll
  for (int g = 0; g < 8; ++g) {
    const int c = 64 * wg + 8 * g + 2 * (lane & 3);
    stg64f(base + (size_t)2 * HS * D_ + r0 * D_ + c, accW2[4 * g], accW2[4 * g + 1]);
    stg64f(base + (size_t)2 * HS * D_ + (r0 + 8) * D_ + c, accW2[4 * g + 2], accW2[4 * g + 3]);
  }
}

// ============================================================================================ DX role
TMN_DEVI void input_role(const BwdPar& p, uint8_t* sm, int cta, int tid, int wg, int wtid, int warp, int lane) {
  const uint32_t su = smem_u32(sm);
  uint64_t* bars = reinterpret_cast<uint64_t*>(sm + X_BAR);
  uint64_t* w_full = bars; uint64_t* w_free = bars + 2; uint64_t* in_full = bars + 4; uint64_t* in_free = bars + 6; uint64_t* x_full = bars + 8;
  const int n_local = (p.tiles > cta) ? (p.tiles - cta + NDX - 1) / NDX : 0;
  auto issue_w = [&](uint32_t seq) {                      // chunk seq % NCH into slot seq & 1
    const int s = seq & 1, j = (int)(seq % NCH); uint8_t* b = sm + X_RING + s * X_SLOT;
    mbar_arrive_expect_tx(w_full + s, 49152);
    tma_load_2d(b, p.ws, w_full + s, j * HS, 0);
#pragma unroll
    for (int c = 0; c < 2; ++c) {
      tma_load_2d(b + 16384 + c * 16384, p.wa, w_full + s, c * 64, j * HS);
      tma_load_2d(b + 16384 + c * 16384 + 8192, p.wb, w_full + s, c * 64, j * HS);
    }
  };
  auto issue_in = [&](int i) {
    const int buf = i & 1, row = (cta + i * NDX) * ROWS; uint8_t* b = sm + X_IN + buf * X_INB;
    mbar_arrive_expect_tx(in_full + buf, 65536);
#pragma unroll
    for (int c = 0; c < 2; ++c)
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        tma_load_2d(b + c * 16384 + h * 8192, p.dy, in_full + buf, c * 64, row + h * 64);
        tma_load_2d(b + X_XN + c * 16384 + h * 8192, p.xn, in_full + buf, c * 64, row + h * 64);
      }
  };
  if (tid == 0) {
    mbar_init(w_full, 1); mbar_init(w_full + 1, 1); mbar_init(w_free, 2); mbar_init(w_free + 1, 2);
    mbar_init(in_full, 1); mbar_init(in_full + 1, 1); mbar_init(in_free, 2); mbar_init(in_free + 1, 2);
    mbar_init(x_full, 1);
    fence_barrier_init();
  }
  float* const dgw = p.dgbw + (size_t)cta * 8 * 256;      // this CTA's eight private rows
  float ln_run=0.f;
  if (tid < 128) reinterpret_cast<float*>(sm + X_GAM)[tid] = p.gamma[tid];
  __syncthreads();
  const uint32_t wmax = (uint32_t)n_local * NCH;
  uint32_t wseq = 0;                                       // chunks consumed
  if (tid == 0) { if (wmax > 0) issue_w(0); if (wmax > 1) issue_w(1); for (int i = 0; i < 2 && i < n_local; ++i) issue_in(i); }
  for (int i = 0; i < n_local; ++i) {
    const int buf = i & 1, trow = (cta + i * NDX) * ROWS;
    const uint32_t inu = su + X_IN + buf * X_INB;
    mbar_wait(in_full + buf, (i >> 1) & 1);
    float acc2[64];
#pragma unroll
    for (int e = 0; e < 64; ++e) acc2[e] = 0.f;
    for (int j = 0; j < NCH; ++j, ++wseq) {
      const int s = (int)(wseq & 1);
      const uint32_t slot = su + X_RING + s * X_SLOT;
      mbar_wait(w_full + s, (uint32_t)((wseq >> 1) & 1));
      float acc[32], AB[64];
#pragma unroll
      for (int e = 0; e < 32; ++e) acc[e] = 0.f;
#pragma unroll
      for (int e = 0; e < 64; ++e) AB[e] = 0.f;
      fence_regs(acc); fence_regs(AB); fence_regs(acc2); wgmma_fence();
#pragma unroll
      for (int ks = 0; ks < 8; ++ks) mma64<0, 1>(acc, dk128(inu + wg * 8192, ks), dmn(slot, ks, 16), ks > 0);
      wgmma_commit();
#pragma unroll
      for (int ks = 0; ks < 8; ++ks) mma128<0, 0>(AB, dk128(inu + X_XN + wg * 8192, ks), dk128(slot + 16384, ks), ks > 0);
      wgmma_commit();
      wgmma_wait<1>(); fence_regs(acc); fence_regs(acc2);   // also retires the previous chunk's d_xn update
      if (j > 0 || i > 0) {                                 // the previous chunk's slot is free
        const int ps = (int)((wseq - 1) & 1);
        if (wtid == 0) mbar_arrive(w_free + ps);
        if (tid == 0) { mbar_wait(w_free + ps, (uint32_t)(((wseq - 1) >> 1) & 1)); if (wseq + 1 < wmax) issue_w(wseq + 1); }
      }
      uint32_t dhp[16];
#pragma unroll
      for (int e = 0; e < 16; ++e) dhp[e] = pack_bf16(acc[2 * e], acc[2 * e + 1]);
      wgmma_wait<0>(); fence_regs(AB);
      if (j == NCH - 1) named_bar_sync(1, 256);             // both warpgroups have retired their a / b GEMMs on this buffer's xn
      if (j == NCH - 1 && tid == 0) {                       // the xn half of this buffer is dead: reload it with x for the epilogue
        mbar_arrive_expect_tx(x_full, 32768);
#pragma unroll
        for (int c = 0; c < 2; ++c)
#pragma unroll
          for (int h = 0; h < 2; ++h) tma_load_2d(sm + X_IN + buf * X_INB + X_XN + c * 16384 + h * 8192, p.x, x_full, c * 64, trow + h * 64);
      }
      // ---- gate straight into the wgmma A-fragment layout: C group g of m64n64 -> A[g >> 1][(g & 1) ? 2 : 0] and +1
      uint32_t fab[8][4];                       // [dA | dB] as one m64k128 A fragment: k-steps 0..3 are dA, 4..7 are dB
#pragma unroll
      for (int g = 0; g < 8; ++g) {
        const int e0 = 4 * g, ks = g >> 1, o = (g & 1) ? 2 : 0;
#pragma unroll
        for (int r = 0; r < 2; ++r) {
          const float a0 = AB[e0 + 2 * r], a1 = AB[e0 + 2 * r + 1];
          const float b0 = AB[e0 + 32 + 2 * r], b1 = AB[e0 + 32 + 2 * r + 1];
          const float g0 = bf16lo(dhp[2 * g + r]), g1 = bf16hi(dhp[2 * g + r]);
          const float s0 = sigmoid_kit(a0), s1 = sigmoid_kit(a1), l0 = a0 * s0, l1 = a1 * s1;
          fab[ks][o + r] = pack_bf16((g0 * b0) * (s0 + l0 * (1.f - s0)), (g1 * b1) * (s1 + l1 * (1.f - s1)));
          fab[ks + 4][o + r] = pack_bf16(g0 * l0, g1 * l1);
        }
      }
      fence_regs(acc2); wgmma_fence();
      {
        const uint64_t bd = dsc_(slot + 16384, 16384, 1024);             // the packed tile is [K = 128][N = 128] for this product
        const uint32_t blo = (uint32_t)bd, bhi = (uint32_t)(bd >> 32);
#pragma unroll
        for (int ks = 0; ks < 8; ++ks) mma128_rs(acc2, fab[ks], blo, bhi, (uint32_t)(ks * 2048) >> 4, 1);
      }
      wgmma_commit();
    }
    wgmma_wait<0>(); fence_regs(acc2);          // the last chunk's slot is released by the next tile's j == 0 (one release per chunk)
    // ---- epilogue: LayerNorm backward + residual + dx, in the m64n128 C-fragment layout (a quad holds one whole row)
    mbar_wait(x_full, i & 1);
    float dgp[32], dbp[32];                                      // this tile's dgamma / dbeta partials (the gate accumulators are dead here)
#pragma unroll
    for (int e = 0; e < 32; ++e) { dgp[e] = 0.f; dbp[e] = 0.f; }
    const int lrow = wg * WGR + 16 * warp + (lane >> 2);          // row within the 128-row tile (and + 8)
    const uint32_t xbase = su + X_IN + buf * X_INB + X_XN, dybase = su + X_IN + buf * X_INB;
#pragma unroll
    for (int rb = 0; rb < 2; ++rb) {
      const int r = lrow + 8 * rb, grow = trow + r;
      const float rs = p.rstd[grow], mean = p.c1[grow] / rs;
      const float* gam = reinterpret_cast<const float*>(sm + X_GAM);
      float ca = 0.f, cb = 0.f;
#pragma unroll                                                   // pass 1: the two row reductions (a quad holds the whole row)
      for (int g = 0; g < 16; ++g) {
        const int col = 8 * g + 2 * (lane & 3);
        const uint32_t xv = lds32(xbase + (col >> 6) * 16384 + swz128((uint32_t)r, (uint32_t)((col & 63) * 2)));
        const uint32_t dn = pack_bf16(acc2[4 * g + 2 * rb], acc2[4 * g + 2 * rb + 1]);        // d_xn rounded once to bf16
        const float n0 = bf16lo(dn), n1 = bf16hi(dn);
        const float x0 = (bf16lo(xv) - mean) * rs, x1 = (bf16hi(xv) - mean) * rs;
        const float w0 = gam[col] * n0, w1 = gam[col + 1] * n1;
        ca += x0 * w0 + x1 * w1; cb += w0 + w1;
        dgp[2 * g] += n0 * x0; dgp[2 * g + 1] += n1 * x1; dbp[2 * g] += n0; dbp[2 * g + 1] += n1;
      }
      ca = quad_sum(ca) * (1.f / D_); cb = quad_sum(cb) * (1.f / D_);
#pragma unroll                                                   // pass 2: dx = bf16(bf16((wdy - xhat ca - cb) rstd) + dy)
      for (int g = 0; g < 16; ++g) {
        const int col = 8 * g + 2 * (lane & 3);
        const uint32_t xv = lds32(xbase + (col >> 6) * 16384 + swz128((uint32_t)r, (uint32_t)((col & 63) * 2)));
        const uint32_t dyv = lds32(dybase + (col >> 6) * 16384 + swz128((uint32_t)r, (uint32_t)((col & 63) * 2)));
        const uint32_t dn = pack_bf16(acc2[4 * g + 2 * rb], acc2[4 * g + 2 * rb + 1]);
        const float x0 = (bf16lo(xv) - mean) * rs, x1 = (bf16hi(xv) - mean) * rs;
        const float w0 = gam[col] * bf16lo(dn), w1 = gam[col + 1] * bf16hi(dn);
        const uint32_t o = pack_bf16((w0 - (x0 * ca + cb)) * rs, (w1 - (x1 * ca + cb)) * rs);
        stg32u(p.dx + (size_t)grow * D_ + col, pack_bf16(bf16lo(o) + bf16lo(dyv), bf16hi(o) + bf16hi(dyv)));
      }
    }
    // ---- dgamma / dbeta of this tile: the eight rows a warp holds reduce over lanes 4 / 8 / 16, then lanes 0-3 add into shared memory
#pragma unroll
    for (int sh = 4; sh < 32; sh <<= 1) {
#pragma unroll
      for (int e = 0; e < 32; ++e) { dgp[e] += __shfl_xor_sync(0xffffffffu, dgp[e], sh); dbp[e] += __shfl_xor_sync(0xffffffffu, dbp[e], sh); }
    }
    // Current xn/x storage is dead after this CTA has completed its LN epilogue.
    __syncthreads();
    float* tmp=reinterpret_cast<float*>(sm+X_IN+buf*X_INB+X_XN);
    if(lane<4){float* row=tmp+(wg*4+warp)*256;
      #pragma unroll
      for(int g=0;g<16;++g){int col=8*g+2*lane;
        row[col]=dgp[2*g];row[col+1]=dgp[2*g+1];row[128+col]=dbp[2*g];row[128+col+1]=dbp[2*g+1];
      }
    }
    __syncthreads();
    float tile_sum=0.f;
    #pragma unroll
    for(int w=0;w<8;++w)tile_sum+=tmp[w*256+tid];
    ln_run+=tile_sum;
    __syncthreads();
    if (wtid == 0) mbar_arrive(in_free + buf);
    if (tid == 0 && i + 2 < n_local) { mbar_wait(in_free + buf, (i >> 1) & 1); issue_in(i + 2); }
  }
  dgw[tid]=ln_run;
}

extern "C" __global__ void __launch_bounds__(256, 1)
transition_bwd_fused(const __grid_constant__ CUtensorMap mdy, const __grid_constant__ CUtensorMap mxn, const __grid_constant__ CUtensorMap mx,
                 const __grid_constant__ CUtensorMap mws, const __grid_constant__ CUtensorMap mwa, const __grid_constant__ CUtensorMap mwb,
                 const float* __restrict__ rstd, const float* __restrict__ c1, const float* __restrict__ gamma,
                 __nv_bfloat16* __restrict__ dx, float* __restrict__ dgam, float* __restrict__ dbeta, float* __restrict__ partw,
                 float* __restrict__ dgbw, int M, int tiles) {
  const BwdPar p{&mdy, &mxn, &mx, &mws, &mwa, &mwb, rstd, c1, gamma, dx, dgam, dbeta, partw, dgbw, M, tiles};
  extern __shared__ __align__(1024) uint8_t sm[];
  const int tid = threadIdx.x, wg = tid >> 7, wtid = tid & 127, warp = wtid >> 5, lane = tid & 31;
  if (blockIdx.x < NDW) weight_role(p, sm, blockIdx.x, tid, wg, wtid, warp, lane);
  else input_role(p, sm, blockIdx.x - NDW, tid, wg, wtid, warp, lane);
}

// partw [NDW][3][64 hs][128 d] fp32 -> bf16 dWa [512][128], dWb [512][128], dWs [128][512]
extern "C" __global__ void reduce_partials(const float* __restrict__ ws, __nv_bfloat16* __restrict__ dWa, __nv_bfloat16* __restrict__ dWb, __nv_bfloat16* __restrict__ dWs,
                                     const float* __restrict__ dgbw, float* __restrict__ dgam, float* __restrict__ dbeta) {
  const int tid=threadIdx.x;
  const int idx = blockIdx.x * blockDim.x + tid;
  if (blockIdx.x >= 768) {
    const int c=(blockIdx.x-768)*32+(tid&31), part=tid>>5;
    float g=0.f,b=0.f;
    for(int r=part;r<NDX;r+=8){g+=dgbw[(size_t)r*2048+c];b+=dgbw[(size_t)r*2048+128+c];}
    __shared__ float scratch[512];scratch[tid]=g;scratch[256+tid]=b;__syncthreads();
    if(tid<32){float gs=0.f,bs=0.f;
      #pragma unroll
      for(int p=0;p<8;++p){gs+=scratch[p*32+tid];bs+=scratch[256+p*32+tid];}
      dgam[c]=gs;dbeta[c]=bs;
    }return;
  }
  if (blockIdx.x >= 512) {
    // Reduce a 16x16 tile, then transpose in shared memory for coalesced dWs stores.
    const int tile=blockIdx.x-512, h0=(tile/8)*16, d0=(tile%8)*16;
    const int h=h0+tid/16, d=d0+tid%16, slice=h/64, hs=h%64;
    float v=0.f;
    for(int r=0;r<DW_REPL;++r)v+=ws[((size_t)(r*8+slice)*3+2)*HS*D_+hs*D_+d];
    __shared__ float transpose[16][17];
    transpose[tid/16][tid%16]=v;__syncthreads();
    dWs[(size_t)(d0+tid/16)*H_+h0+tid%16]=__float2bfloat16_rn(transpose[tid%16][tid/16]);
    return;
  }
  const int which = idx / (8 * HS * D_), rem = idx % (8 * HS * D_), slice = rem / (HS * D_), hs = (rem / D_) % HS, d = rem % D_;
  float v = 0.f;
  for (int r = 0; r < DW_REPL; ++r) v += ws[((size_t)(r * 8 + slice) * 3 + which) * HS * D_ + hs * D_ + d];
  const __nv_bfloat16 o = __float2bfloat16_rn(v);
  if (which == 0) dWa[(slice * HS + hs) * D_ + d] = o;
  else if (which == 1) dWb[(slice * HS + hs) * D_ + d] = o;
  else dWs[(size_t)d * H_ + slice * HS + hs] = o;
}

