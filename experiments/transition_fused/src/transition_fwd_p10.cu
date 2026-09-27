// transition_fwd.cu — the Transition forward (LayerNorm + SwiGLU expand + squeeze + residual) of the MiniWorld pair
// Transition at D = 128, H = 4D = 512, bf16, as ONE fused sm_90a kernel that also emits what the backward needs.
// SPDX-License-Identifier: Apache-2.0
// Built on the Anthropic native v5 device primitives (vendor/anthropic_v5/csrc), and structured like the input role of
// transition_bwd.cu: a persistent CTA per 128-row tile, a two-slot TMA weight ring over the eight 64-unit hidden chunks,
// and the SwiGLU result formed as bf16 straight in the wgmma A-fragment registers.
//
//   xn = bf16(LN(x))                                     saved, with rstd and c1 = mean rstd, for the backward
//   per hidden chunk j:  [a|b] = xn [Wa_j; Wb_j]^T       one m64n128 chain over the packed weight tile
//                        h_j   = bf16(silu(a) b)         in the m64k64 A-fragment layout, no shared-memory round trip
//                        acc  += h_j Ws^T_j              RS m64n128, accumulating in registers across all eight chunks
//   out = bf16(x + acc)
//
// The path it replaces runs three kernels and puts the [M][512] SwiGLU activation through HBM twice (151 MB each way at
// L384); here it never leaves registers, and the only traffic is x in, xn out and out out.
//
// p10 = p9 with G1 descriptors as (lo, hi) + immediate offsets and gamma/beta re-read per tile (register pressure).
// p9: software-pipelined chunk loop.  The base kernel runs  G1(j) -> wait<0> -> SwiGLU(j) -> G2(j)  per chunk, so the tensor
// pipe is idle for every SwiGLU.  Here G1(j+1) is issued BEFORE SwiGLU(j) into a second [a|b] accumulator, so the tensor pipe
// computes chunk j+1's expand while the SM forms chunk j's h.  That needs the [Wa;Wb] tile of j+1 resident while Ws^T_j is
// still being read, so the weight ring is split in two: [Wa;Wb] 2 x 32 KB, Ws^T 3 x 16 KB (G2(j) retires one chunk later
// than before, so the squeeze operand needs one more slot of slack).  The room comes from single-buffering xn: each
// warpgroup reads and rewrites only its own 64 rows, and every wgmma reading them has retired before the next LayerNorm.
#include "tmn_kernels.cuh"
using namespace tmn; using namespace tmn::sm90;

#ifndef NCTA
#define NCTA 132
#endif
#ifndef FWD_SAVE
#define FWD_SAVE 1          // 1: also write xn, rstd and c1 (what the backward needs). 0: inference, output only.
#endif
constexpr int D_ = 128, H_ = 512, HS = 64, NCH = H_ / HS, ROWS = 128, WGR = 64;

// ---- shared memory: split weight ring ([Wa;Wb] x NAB, Ws^T x NWS), double-buffered x, single xn
constexpr int NAB = 2, F_ABS = 32768, F_AB = 0;                // [Wa_j; Wb_j] packed, [128 n][128 d]
constexpr int NWS = 3, F_WSS = 16384, F_WS = F_AB + NAB * F_ABS;   // Ws^T_j, [64 hs][128 d]
constexpr int F_X = F_WS + NWS * F_WSS, F_XB = 32768;          // x  tile [128 rows][128 d] bf16, double-buffered
constexpr int F_XN = F_X + 2 * F_XB;                           // xn tile, single
constexpr int F_BAR = F_XN + F_XB;
static_assert(F_WS % 1024 == 0 && F_X % 1024 == 0 && F_XN % 1024 == 0, "128B-swizzled tiles need 1 KB alignment");
constexpr int SMEM_BYTES = 231424;
static_assert(F_BAR + 256 <= SMEM_BYTES, "shared memory budget");

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


template <int FIRST, int AOFF, int BOFF>
TMN_DEVI void mma128_ss(float (&d)[64], uint32_t alo, uint32_t ahi, uint32_t blo, uint32_t bhi) {
#define TMN_D64 "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}"
#define TMN_OUT64(C) C(d[0]), C(d[1]), C(d[2]), C(d[3]), C(d[4]), C(d[5]), C(d[6]), C(d[7]), C(d[8]), C(d[9]), C(d[10]), C(d[11]), C(d[12]), C(d[13]), C(d[14]), C(d[15]), C(d[16]), C(d[17]), C(d[18]), C(d[19]), C(d[20]), C(d[21]), C(d[22]), C(d[23]), C(d[24]), C(d[25]), C(d[26]), C(d[27]), C(d[28]), C(d[29]), C(d[30]), C(d[31]), C(d[32]), C(d[33]), C(d[34]), C(d[35]), C(d[36]), C(d[37]), C(d[38]), C(d[39]), C(d[40]), C(d[41]), C(d[42]), C(d[43]), C(d[44]), C(d[45]), C(d[46]), C(d[47]), C(d[48]), C(d[49]), C(d[50]), C(d[51]), C(d[52]), C(d[53]), C(d[54]), C(d[55]), C(d[56]), C(d[57]), C(d[58]), C(d[59]), C(d[60]), C(d[61]), C(d[62]), C(d[63])
#define TMN_PF(x) "+f"(x)
#define TMN_WF(x) "=f"(x)
  if constexpr (FIRST) {
    asm volatile("{\n .reg .b32 la, lb;\n .reg .b64 da, db;\n add.u32 la, %64, %68;\n add.u32 lb, %66, %69;\n"
                 " mov.b64 da, {la, %65};\n mov.b64 db, {lb, %67};\n"
                 "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 " TMN_D64 ", da, db, 0, 1, 1, 0, 0;\n}\n"
                 : TMN_OUT64(TMN_WF) : "r"(alo), "r"(ahi), "r"(blo), "r"(bhi), "n"(AOFF), "n"(BOFF));
  } else {
    asm volatile("{\n .reg .b32 la, lb;\n .reg .b64 da, db;\n add.u32 la, %64, %68;\n add.u32 lb, %66, %69;\n"
                 " mov.b64 da, {la, %65};\n mov.b64 db, {lb, %67};\n"
                 "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 " TMN_D64 ", da, db, 1, 1, 1, 0, 0;\n}\n"
                 : TMN_OUT64(TMN_PF) : "r"(alo), "r"(ahi), "r"(blo), "r"(bhi), "n"(AOFF), "n"(BOFF));
  }
#undef TMN_D64
#undef TMN_OUT64
#undef TMN_PF
#undef TMN_WF
}
// the eight K-steps of G1: dk128's address advance, (ks >> 2) * 16384 + (ks & 3) * 32 bytes, in descriptor units (>> 4)
template <int KS> struct K128 { static constexpr int off = ((KS >> 2) * 16384 + (KS & 3) * 32) >> 4; };
template <int KS = 0>
TMN_DEVI void g1_chain(float (&d)[64], uint32_t alo, uint32_t ahi, uint32_t blo, uint32_t bhi) {
  mma128_ss<KS == 0, K128<KS>::off, K128<KS>::off>(d, alo, ahi, blo, bhi);
  if constexpr (KS + 1 < 8) g1_chain<KS + 1>(d, alo, ahi, blo, bhi);
}


struct Par {                                 // the tensor maps stay in the grid-constant parameter bank
  const CUtensorMap *x, *wa, *wb, *wst, *outm;
  const float *gamma, *beta;
  __nv_bfloat16 *xn, *out;
  float *rstd, *c1;
  int M, tiles;
};

TMN_DEVI void tma_store_2d(const CUtensorMap* map, const void* src, int c0, int c1) {
  asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%2, %3}], [%1];"
               :: "l"(map), "r"(smem_u32(src)), "r"(c0), "r"(c1) : "memory");
}
TMN_DEVI uint2 lds64u(uint32_t a) { uint2 v; asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];" : "=r"(v.x), "=r"(v.y) : "r"(a) : "memory"); return v; }
TMN_DEVI void stg64u(void* p, uint32_t a, uint32_t b) { asm volatile("st.global.v2.b32 [%0], {%1,%2};" :: "l"(p), "r"(a), "r"(b) : "memory"); }
TMN_DEVI float warp_allsum(float v) {
#pragma unroll
  for (int k = 16; k; k >>= 1) v += __shfl_xor_sync(0xffffffffu, v, k);
  return v;
}

extern "C" __global__ void __launch_bounds__(256, 1)
transition_fwd_fused(const __grid_constant__ CUtensorMap mx, const __grid_constant__ CUtensorMap mwa,
                     const __grid_constant__ CUtensorMap mwb, const __grid_constant__ CUtensorMap mwst,
                     const __grid_constant__ CUtensorMap mout, const float* __restrict__ gamma, const float* __restrict__ beta,
                     __nv_bfloat16* __restrict__ xn, __nv_bfloat16* __restrict__ out,
                     float* __restrict__ rstd, float* __restrict__ c1, int M, int tiles, float eps) {
  const Par p{&mx, &mwa, &mwb, &mwst, &mout, gamma, beta, xn, out, rstd, c1, M, tiles};
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  const int tid = threadIdx.x, wg = tid >> 7, wtid = tid & 127, warp = wtid >> 5, lane = tid & 31;
  const int cta = blockIdx.x;
  uint64_t* bars = reinterpret_cast<uint64_t*>(sm + F_BAR);
  uint64_t* ab_full = bars; uint64_t* ab_free = bars + NAB;
  uint64_t* ws_full = bars + 2 * NAB; uint64_t* ws_free = ws_full + NWS;
  uint64_t* x_full = ws_free + NWS; uint64_t* x_free = x_full + 2;
  const int n_local = (tiles > cta) ? (tiles - cta + NCTA - 1) / NCTA : 0;
  const uint32_t wmax = (uint32_t)n_local * NCH;
  auto issue_ab = [&](uint32_t seq) {                     // [Wa_j; Wb_j] of chunk seq % NCH into slot seq % NAB
    const int s = (int)(seq % NAB), j = (int)(seq % NCH); uint8_t* b = sm + F_AB + s * F_ABS;
    mbar_arrive_expect_tx(ab_full + s, F_ABS);
#pragma unroll
    for (int c = 0; c < 2; ++c) {
      tma_load_2d(b + c * 16384, p.wa, ab_full + s, c * 64, j * HS);           // rows 0..63 of the packed tile
      tma_load_2d(b + c * 16384 + 8192, p.wb, ab_full + s, c * 64, j * HS);    // rows 64..127
    }
  };
  auto issue_ws = [&](uint32_t seq) {                     // Ws^T_j of chunk seq % NCH into slot seq % NWS
    const int s = (int)(seq % NWS), j = (int)(seq % NCH); uint8_t* b = sm + F_WS + s * F_WSS;
    mbar_arrive_expect_tx(ws_full + s, F_WSS);
#pragma unroll
    for (int c = 0; c < 2; ++c) tma_load_2d(b + c * 8192, p.wst, ws_full + s, c * 64, j * HS);
  };
  auto issue_x = [&](int i) {
    const int buf = i & 1, row = (cta + i * NCTA) * ROWS; uint8_t* b = sm + F_X + buf * F_XB;
    mbar_arrive_expect_tx(x_full + buf, 32768);
#pragma unroll
    for (int c = 0; c < 2; ++c)
#pragma unroll
      for (int h = 0; h < 2; ++h) tma_load_2d(b + c * 16384 + h * 8192, p.x, x_full + buf, c * 64, row + h * 64);
  };
  if (tid == 0) {
    for (int s = 0; s < NAB; ++s) { mbar_init(ab_full + s, 1); mbar_init(ab_free + s, 2); }
    for (int s = 0; s < NWS; ++s) { mbar_init(ws_full + s, 1); mbar_init(ws_free + s, 2); }
    mbar_init(x_full, 1); mbar_init(x_full + 1, 1); mbar_init(x_free, 2); mbar_init(x_free + 1, 2);
    fence_barrier_init();
  }
  __syncthreads();
  if (tid == 0) {
    for (uint32_t q = 0; q < NAB && q < wmax; ++q) issue_ab(q);
    for (uint32_t q = 0; q < NWS && q < wmax; ++q) issue_ws(q);
    for (int i = 0; i < 2 && i < n_local; ++i) issue_x(i);
  }
  // LayerNorm mapping: this thread owns columns 4 lane .. 4 lane + 3 of rows 16 warp + ii (ii < 16) of its warpgroup's half
  const int c0 = 4 * lane;
  const uint32_t xcol = (uint32_t)(c0 >> 6) * 16384 + (uint32_t)(((c0 & 63) * 2) & 15);
  const uint32_t gran = (uint32_t)((c0 & 63) * 2) >> 4;

  for (int i = 0; i < n_local; ++i) {
    const int buf = i & 1, trow = (cta + i * NCTA) * ROWS;
    const uint32_t xu = su + F_X + buf * F_XB, xnu = su + F_XN;
    mbar_wait(x_full + buf, (i >> 1) & 1);
    float g4[4], b4[4];                                   // re-read per tile (L1 hit): 8 registers not held across the chunk loop
    { const float4 gg = __ldg(reinterpret_cast<const float4*>(p.gamma + c0)), bbv = __ldg(reinterpret_cast<const float4*>(p.beta + c0));
      g4[0] = gg.x; g4[1] = gg.y; g4[2] = gg.z; g4[3] = gg.w; b4[0] = bbv.x; b4[1] = bbv.y; b4[2] = bbv.z; b4[3] = bbv.w; }
    // ---------------------------------------------------------------- LayerNorm, in registers, into the xn tile
    // Eight rows at a time: the two warp reductions a row needs are five dependent shuffles each, and one row at a time
    // left that latency fully exposed (34 us of 145 by ablation). Eight independent chains per step hide it; the reduction
    // order within a row is unchanged, so the statistics are bit-identical to the serial form.
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      uint2 v[8];
      uint32_t ad[8];
      float acc8[8];
#pragma unroll
      for (int u = 0; u < 8; ++u) {
        const int r = wg * WGR + 16 * warp + 8 * half + u;
        ad[u] = xu + xcol + (uint32_t)r * 128u + ((gran ^ ((uint32_t)r & 7u)) << 4);
        v[u] = lds64u(ad[u]);
      }
#pragma unroll
      for (int u = 0; u < 8; ++u)
        acc8[u] = (bf16lo(v[u].x) + bf16hi(v[u].x)) + (bf16lo(v[u].y) + bf16hi(v[u].y));
#pragma unroll
      for (int k = 16; k; k >>= 1) {
#pragma unroll
        for (int u = 0; u < 8; ++u) acc8[u] += __shfl_xor_sync(0xffffffffu, acc8[u], k);
      }
      float mean8[8];
#pragma unroll
      for (int u = 0; u < 8; ++u) { mean8[u] = acc8[u] * (1.f / D_); acc8[u] = 0.f; }
#pragma unroll
      for (int u = 0; u < 8; ++u) {
        float d;
        d = bf16lo(v[u].x) - mean8[u]; acc8[u] += d * d;
        d = bf16hi(v[u].x) - mean8[u]; acc8[u] += d * d;
        d = bf16lo(v[u].y) - mean8[u]; acc8[u] += d * d;
        d = bf16hi(v[u].y) - mean8[u]; acc8[u] += d * d;
      }
#pragma unroll
      for (int k = 16; k; k >>= 1) {
#pragma unroll
        for (int u = 0; u < 8; ++u) acc8[u] += __shfl_xor_sync(0xffffffffu, acc8[u], k);
      }
#pragma unroll
      for (int u = 0; u < 8; ++u) {
        const int r = wg * WGR + 16 * warp + 8 * half + u;
        const float mean = mean8[u], rs = rsqrtf(acc8[u] * (1.f / D_) + eps);
        uint2 o;
        o.x = pack_bf16((bf16lo(v[u].x) - mean) * rs * g4[0] + b4[0], (bf16hi(v[u].x) - mean) * rs * g4[1] + b4[1]);
        o.y = pack_bf16((bf16lo(v[u].y) - mean) * rs * g4[2] + b4[2], (bf16hi(v[u].y) - mean) * rs * g4[3] + b4[3]);
        asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(xnu + xcol + (uint32_t)r * 128u + ((gran ^ ((uint32_t)r & 7u)) << 4)),
                     "r"(o.x), "r"(o.y) : "memory");
#if FWD_SAVE
        stg64u(p.xn + (size_t)(trow + r) * D_ + c0, o.x, o.y);   // the backward's saved xn: 32 lanes x 8 B = one whole row
        if (lane == 0) { const int gr = trow + r; p.rstd[gr] = rs; p.c1[gr] = mean * rs; }
#endif
      }
    }
    fence_proxy_async();                                  // the generic stores of xn -> visible to the wgmma operand reads
    named_bar_sync(1 + wg, 128);
    // ---------------------------------------------------------------- the eight hidden chunks, software-pipelined
    // Commit order per tile: G1(0) | G1(1) G2(0) | G1(2) G2(1) | ... | G1(7) G2(6) | G2(7).  Chunk j retires G1(j) with the
    // groups committed after it still in flight: G2(j-1) and G1(j+1) where they exist -> wait<2>, or wait<1> at j = 0 and
    // j = 7.  Retiring G1(j) also retires G2(j-2), which is when that chunk's Ws^T slot is released.
    float acc[64];
#pragma unroll
    for (int e = 0; e < 64; ++e) acc[e] = 0.f;
    float A0[64], A1[64];
    const uint32_t cb = (uint32_t)i * NCH;
    const uint64_t xd = dk128(xnu + wg * 8192, 0);
    const uint32_t xd_lo = (uint32_t)xd, xd_hi = (uint32_t)(xd >> 32);
    auto issue_g1 = [&](float (&AB)[64], uint32_t c) __attribute__((always_inline)) {
      const int s = (int)(c % NAB);
      mbar_wait(ab_full + s, (c / NAB) & 1u);
      fence_regs(acc); wgmma_fence();                    // AB is write-only at k-step 0 (scale-d = 0): no zeroing, no fence on it
      const uint64_t bd = dk128(su + F_AB + s * F_ABS, 0);
      g1_chain(AB, xd_lo, xd_hi, (uint32_t)bd, (uint32_t)(bd >> 32));
      wgmma_commit();
    };
    auto chunk = [&](float (&ABc)[64], float (&ABn)[64], int j) __attribute__((always_inline)) {
      const uint32_t c = cb + (uint32_t)j;
      if (j + 1 < NCH) issue_g1(ABn, c + 1);             // the tensor pipe expands chunk j+1 while this chunk's SwiGLU runs
      if (j == 0 || j == NCH - 1) wgmma_wait<1>(); else wgmma_wait<2>();
      fence_regs(ABc);
      if (wtid == 0) {
        mbar_arrive(ab_free + (int)(c % NAB));           // G1(c) has retired: its [Wa;Wb] slot may be refilled
        if (c >= 2) mbar_arrive(ws_free + (int)((c - 2) % NWS));   // so has G2(c-2): its Ws^T slot too
      }
      uint32_t fh[4][4];                                  // h = bf16(silu(a) b): C group g holds a, group g + 8 holds b
#pragma unroll
      for (int g = 0; g < 8; ++g) {
        const int e0 = 4 * g, ks = g >> 1, o = (g & 1) ? 2 : 0;
#pragma unroll
        for (int r = 0; r < 2; ++r) {
          const float a0 = ABc[e0 + 2 * r], a1 = ABc[e0 + 2 * r + 1];
          const float b0 = ABc[e0 + 32 + 2 * r], b1 = ABc[e0 + 32 + 2 * r + 1];
          fh[ks][o + r] = pack_bf16(a0 * sigmoid_kit(a0) * b0, a1 * sigmoid_kit(a1) * b1);
        }
      }
      if (tid == 0) {                                     // refills: both warpgroups had this chunk's SwiGLU to reach their release
        if (c + NAB < wmax) { mbar_wait(ab_free + (int)(c % NAB), (c / NAB) & 1u); issue_ab(c + NAB); }
        if (c >= 2 && c + 1 < wmax) { mbar_wait(ws_free + (int)((c - 2) % NWS), ((c - 2) / NWS) & 1u); issue_ws(c + 1); }
      }
      const int s = (int)(c % NWS);
      mbar_wait(ws_full + s, (c / NWS) & 1u);
      fence_regs(acc); wgmma_fence();
      {
        const uint64_t bd = dsc_(su + F_WS + s * F_WSS, 8192, 1024);   // Ws^T_j is [K = 64 hs][N = 128 d]
        const uint32_t blo = (uint32_t)bd, bhi = (uint32_t)(bd >> 32);
#pragma unroll
        for (int ks = 0; ks < 4; ++ks) mma128_rs(acc, fh[ks], blo, bhi, (uint32_t)(ks * 2048) >> 4, 1);
      }
      wgmma_commit();
    };
    issue_g1(A0, cb);
#pragma unroll
    for (int jj = 0; jj < NCH; jj += 2) { chunk(A0, A1, jj); chunk(A1, A0, jj + 1); }
    wgmma_wait<0>(); fence_regs(acc);
    // ---------------------------------------------------------------- out = bf16(x + acc), added IN PLACE over the x tile and
    // handed to one TMA store per warpgroup half.  Straight 4-byte global stores from this fragment layout touch eight
    // half-used 32-byte sectors per instruction and measured 60 us of the kernel's 145.
    const int lrow = wg * WGR + 16 * warp + (lane >> 2);
#pragma unroll
    for (int rb = 0; rb < 2; ++rb) {
      const int r = lrow + 8 * rb;
#pragma unroll
      for (int g = 0; g < 16; ++g) {
        const int col = 8 * g + 2 * (lane & 3);
        const uint32_t ad = xu + (col >> 6) * 16384 + swz128((uint32_t)r, (uint32_t)((col & 63) * 2));
        const uint32_t xv = lds32(ad);
        sts32(ad, pack_bf16(bf16lo(xv) + acc[4 * g + 2 * rb], bf16hi(xv) + acc[4 * g + 2 * rb + 1]));
      }
    }
    fence_proxy_async();
    named_bar_sync(1 + wg, 128);
    if (wtid == 0) {
#pragma unroll
      for (int c = 0; c < 2; ++c) tma_store_2d(p.outm, sm + F_X + buf * F_XB + c * 16384 + wg * 8192, c * 64, trow + wg * WGR);
      tma_store_commit();
      tma_store_wait_all();                                // the store has read the tile: the buffer may be refilled
      mbar_arrive(x_free + buf);
    }
    if (tid == 0 && i + 2 < n_local) { mbar_wait(x_free + buf, (i >> 1) & 1); issue_x(i + 2); }
  }
}
