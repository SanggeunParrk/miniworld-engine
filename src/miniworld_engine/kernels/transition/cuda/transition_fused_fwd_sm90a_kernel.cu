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
#include "tmn_kernels.cuh"
using namespace tmn; using namespace tmn::sm90;

#ifndef NCTA
#define NCTA 132
#endif
#ifndef FWD_SAVE
#define FWD_SAVE 1          // 1: also write xn, rstd and c1 (what the backward needs). 0: inference, output only -- see below.
#endif
#ifndef MW_N
#define MW_N 4            // hidden width / D: 4 (the AF3 Transition) or 2
#endif
#ifndef RSMASK
#define RSMASK 0xFF
#endif
#ifndef CPX
#define CPX 0   // 1: the first tile's x rows by cp.async (no tensor-map fetch on the critical path), each warpgroup its own rows
#endif
#ifndef STAGGER
#define STAGGER 0   // 1: CTA c walks the hidden chunks starting at chunk c % NCH (spreads the weight fetches over the L2 slices)
#endif
#ifndef DECOUPLE
#define DECOUPLE 0   // 1: the weight-slot refill is issued by whichever warpgroup retires the slot last (no blocking handshake between the two)
#endif
#ifndef XFIRST
#define XFIRST 0   // 1: issue the x tile's TMA before the weight chunks'
#endif
#ifndef SUBCH
#define SUBCH 0   // 1: software-pipelined 32-unit sub-chunks (G1 of sub-chunk t + 1 runs on the tensor core while swiglu of t runs on MUFU)
#endif
#ifndef NEWTON
#define NEWTON 0   // 1: the sigmoid reciprocal by three Newton steps on the FMA pipe instead of MUFU.RCP (the swiglu of a chunk is MUFU-bound)
#endif
#ifndef WSLOTS
#define WSLOTS 2   // weight-ring slots (3 hides the refill latency of chunks 2 and 3 for n = 2)
#endif
#ifndef GBSMEM
#define GBSMEM 0   // 1: gamma/beta staged in shared memory before x lands (the fragment LayerNorm no longer waits on global loads)
#endif
#ifndef TRACE
#define TRACE 0   // 1: record clock64() timestamps of CTA 0 into g_trace (read back with transition_fused_trace)
#endif
#if TRACE
__device__ unsigned long long g_trace[64];
#define TR(i) do { if (cta == 0 && wtid == 0) g_trace[wg * 32 + (i)] = clock64(); } while (0)
#else
#define TR(i) do {} while (0)
#endif
#ifndef SKEW
#define SKEW 0   // 1: warpgroup 1 issues its first first-GEMM after warpgroup 0 (the two then ping-pong MMA against swiglu)
#endif
#ifndef ABL
#define ABL 0   // timing ablation only (wrong results): 1 stop after LN, 2 no epilogue/store, 3 no GEMM chunks
#endif
#ifndef PDL
#define PDL 0   // 1: programmatic dependent launch -- the prologue overlaps the predecessor, `griddepcontrol.wait` guards every read
#endif
#ifndef SPLIT2
#define SPLIT2 0   // 1: the 64-row / split-hidden latency variant (n = 2 inference, one tile per CTA)
#endif
#ifndef RESIDENT
#define RESIDENT 0   // 1: every weight chunk resident (needs NCH <= 4, LNFRAG, and at most one tile per CTA)
#endif
#ifndef LNFRAG
#define LNFRAG 0   // DISABLED: D128 fragment-LN path measured wrong vs fp64
#endif
constexpr int D_ = 128, H_ = MW_N * 128, HS = 64, NCH = H_ / HS, ROWS = 128, WGR = 64;

// ---- shared memory: a two-slot weight ring plus double-buffered x and xn tiles
constexpr int F_SLOT = 49152, F_RING = 0;                      // slot: [Wa_j; Wb_j] 32 KB at +0, then Ws^T_j 16 KB at +32768
constexpr int F_WST = 32768;
constexpr int NSLOT = RESIDENT ? NCH : WSLOTS;                      // weight-ring slots (resident: one per chunk, never recycled)
constexpr int F_X = NSLOT * F_SLOT, F_XB = 32768;                  // x  tile [128 rows][128 d] bf16, double-buffered
constexpr int F_XN = F_X + 2 * F_XB;                           // xn tile, double-buffered
#if RESIDENT
constexpr int F_BAR = F_X + F_XB;                             // single x buffer, no xn tile
#elif LNFRAG == 1 || LNFRAG == 2
constexpr int F_BAR = F_XN;                                    // (no xn tile)
#else
constexpr int F_BAR = F_XN + 2 * F_XB;
#endif
constexpr int SMEM_BYTES = 231424;
static_assert(F_BAR + 1280 <= SMEM_BYTES, "shared memory budget");

TMN_DEVI float warp_sum(float x) {
#pragma unroll
  for (int k = 16; k; k >>= 1) x += __shfl_xor_sync(0xffffffffu, x, k);
  return x;
}
#if NEWTON
TMN_DEVI float rcp_newton(float d) {                  // d in [1, 8.6e37]: integer-subtract seed (<= 10 % off), three Newton steps -> fp32
  float r = __uint_as_float(0x7EF311C7u - __float_as_uint(d));
  r = r * fmaf(-d, r, 2.f); r = r * fmaf(-d, r, 2.f); r = r * fmaf(-d, r, 2.f);
  return r;
}
TMN_DEVI float sigmoid_kit(float a) {
  return rcp_newton(__fadd_rn(1.f, ex2f(fminf(__fmul_rn(-1.4426950408889634f, a), 126.f))));
}
#else
TMN_DEVI float sigmoid_kit(float a) {                 // math::sigmoid of the Anthropic kit: rcp.approx.ftz(1 + ex2.approx.ftz(-a log2 e))
  return rcpf(__fadd_rn(1.f, ex2f(__fmul_rn(-1.4426950408889634f, a))));   // (math::sigmoid); div.full is a multi-instruction sequence
}
#endif
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
// as mma128_rs with B K-major (trans-b = 0): the packed [Wa_j; Wb_j] tile of the first GEMM when xn is an RS operand
TMN_DEVI void mma128_rs_k(float (&d)[64], const uint32_t (&a)[4], uint32_t desc_lo, uint32_t desc_hi, uint32_t off16, int accumulate) {
  asm volatile("{\n .reg .pred p;\n .reg .b32 lo;\n .reg .b64 dsc;\n setp.ne.b32 p, %71, 0;\n add.u32 lo, %68, %70;\n mov.b64 dsc, {lo, %69};\n"
    "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, {%64,%65,%66,%67}, dsc, p, 1, 1, 0;\n}\n"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(desc_lo), "r"(desc_hi), "r"(off16), "r"(accumulate));
}


// m64n64k16 with A from registers and B K-major (trans-b = 0): the [a(s); b(s)] 64-row sub-chunk tile of the first GEMM
TMN_DEVI void mma64_rs_k(float (&d)[32], const uint32_t (&a)[4], uint32_t desc_lo, uint32_t desc_hi, int accumulate) {
  asm volatile("{\n .reg .pred p;\n .reg .b64 dsc;\n setp.ne.b32 p, %38, 0;\n mov.b64 dsc, {%36, %37};\n"
    "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, {%32,%33,%34,%35}, dsc, p, 1, 1, 0;\n}\n"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(desc_lo), "r"(desc_hi), "r"(accumulate));
}

struct FwdPar {                                 // the tensor maps stay in the grid-constant parameter bank
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
                     float* __restrict__ rstd, float* __restrict__ c1, int M, int tiles, float eps, int save,
                     const __nv_bfloat16* __restrict__ xin) {
  const FwdPar p{&mx, &mwa, &mwb, &mwst, &mout, gamma, beta, xn, out, rstd, c1, M, tiles};
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  const int tid = threadIdx.x, wg = tid >> 7, wtid = tid & 127, warp = wtid >> 5, lane = tid & 31;
  const int cta = blockIdx.x;
  uint64_t* bars = reinterpret_cast<uint64_t*>(sm + F_BAR);
  uint64_t* w_full = bars; uint64_t* w_free = bars + NSLOT; uint64_t* x_full = bars + 2 * NSLOT; uint64_t* x_free = bars + 2 * NSLOT + 2;
  uint32_t* w_cnt = reinterpret_cast<uint32_t*>(sm + F_BAR + 64 + 16 * 8);   // [NSLOT] retire counters (DECOUPLE)
  const int n_local = (tiles > cta) ? (tiles - cta + NCTA - 1) / NCTA : 0;
  const uint32_t wmax = (uint32_t)n_local * NCH;
  const bool pp = (DECOUPLE != 0) && (tiles <= 2 * NCTA);   // ping-pong only while a CTA has <= 2 tiles (1 - 3 % slower beyond: the extra coupling is gone)
#if CPX
  if (n_local > 0) {                                       // tile 0, own 64 rows: 8 x 16 B per thread, 128B-swizzled exactly as the TMA lays it out
    const uint32_t xdst = su + F_X;
    const __nv_bfloat16* xsrc = xin + (size_t)(cta * ROWS + wg * WGR) * D_;
#pragma unroll
    for (int u = 0; u < 8; ++u) {
      const int idx = wtid + 128 * u, r = idx >> 4, c = (idx >> 3) & 1, k = idx & 7;
      const uint32_t dst = xdst + (uint32_t)c * 16384u + swz128((uint32_t)(wg * WGR + r), (uint32_t)(k * 16));
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst), "l"(xsrc + (size_t)r * D_ + c * 64 + k * 8) : "memory");
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
  }
#endif
  auto issue_w = [&](uint32_t seq) {                      // chunk seq % NCH into slot seq & 1
    const int s = (int)(seq % NSLOT), j = (STAGGER ? (int)((seq + (uint32_t)cta) % NCH) : (int)(seq % NCH)); uint8_t* b = sm + F_RING + s * F_SLOT;
    mbar_arrive_expect_tx(w_full + s, 49152);
#pragma unroll
    for (int c = 0; c < 2; ++c) {
#if SUBCH
      // 32-row boxes: rows [a(0) 32 | b(0) 32 | a(1) 32 | b(1) 32] so each 32-unit sub-chunk is one contiguous 64-row B tile
#pragma unroll
      for (int sc = 0; sc < 2; ++sc) {
        tma_load_2d(b + c * 16384 + sc * 8192, p.wa, w_full + s, c * 64, j * HS + 32 * sc);
        tma_load_2d(b + c * 16384 + sc * 8192 + 4096, p.wb, w_full + s, c * 64, j * HS + 32 * sc);
      }
#else
      tma_load_2d(b + c * 16384, p.wa, w_full + s, c * 64, j * HS);            // rows 0..63 of the packed [Wa_j; Wb_j] tile
      tma_load_2d(b + c * 16384 + 8192, p.wb, w_full + s, c * 64, j * HS);      // rows 64..127
#endif
      tma_load_2d(b + F_WST + c * 8192, p.wst, w_full + s, c * 64, j * HS);     // Ws^T_j, [64 hs][128 d]
    }
  };
  auto issue_x = [&](int i) {
    const int buf = i & 1, row = (cta + i * NCTA) * ROWS; uint8_t* b = sm + F_X + buf * F_XB;
    mbar_arrive_expect_tx(x_full + buf, 32768);
#pragma unroll
    for (int c = 0; c < 2; ++c)
#pragma unroll
      for (int h = 0; h < 2; ++h) tma_load_2d(b + c * 16384 + h * 8192, p.x, x_full + buf, c * 64, row + h * 64);
  };
  if (tid == 0) {                                          // descriptors first: the five fetches overlap the barrier setup
    tma_prefetch_desc(&mx); tma_prefetch_desc(&mwa); tma_prefetch_desc(&mwb); tma_prefetch_desc(&mwst); tma_prefetch_desc(&mout);
    for (int k = 0; k < NSLOT; ++k) { mbar_init(w_full + k, 1); mbar_init(w_free + k, 2); w_cnt[k] = 0; }
    mbar_init(x_full, 1); mbar_init(x_full + 1, 1); mbar_init(x_free, 2); mbar_init(x_free + 1, 2);
    fence_barrier_init();
  }
  __syncthreads();
#if GBSMEM
  float* gbs = reinterpret_cast<float*>(sm + F_BAR + 128);          // [gamma 128][beta 128]
  gbs[tid] = (tid < 128) ? p.gamma[tid] : p.beta[tid - 128];
#endif
#if PDL
  asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
  asm volatile("griddepcontrol.wait;\n" ::: "memory");        // x, weights and gamma/beta may be the predecessor's output
#endif
  TR(0);
  uint32_t wseq = 0;
  if (tid == 0) {
#if XFIRST
    if (n_local > 0 && !CPX) issue_x(0);                                        // the LayerNorm waits on x, not on the weights
    for (uint32_t k = 0; k < (uint32_t)NSLOT; ++k) if (wmax > k && ABL != 3) issue_w(k);
    if (!RESIDENT && n_local > 1) issue_x(1);
#else
    for (uint32_t k = 0; k < (uint32_t)NSLOT; ++k) if (wmax > k && ABL != 3) issue_w(k);
    for (int i = 0; i < (RESIDENT ? 1 : 2) && i < n_local; ++i) issue_x(i);
#endif
  }
  // LayerNorm mapping: this thread owns columns 4 lane .. 4 lane + 3 of rows 16 warp + ii (ii < 16) of its warpgroup's half
  const int c0 = 4 * lane;
  const uint32_t xcol = (uint32_t)(c0 >> 6) * 16384 + (uint32_t)(((c0 & 63) * 2) & 15);
  const uint32_t gran = (uint32_t)((c0 & 63) * 2) >> 4;
  float g4[4], b4[4];
  { const float4 gg = *reinterpret_cast<const float4*>(p.gamma + c0), bbv = *reinterpret_cast<const float4*>(p.beta + c0);
    g4[0] = gg.x; g4[1] = gg.y; g4[2] = gg.z; g4[3] = gg.w; b4[0] = bbv.x; b4[1] = bbv.y; b4[2] = bbv.z; b4[3] = bbv.w; }

  for (int i = 0; i < n_local; ++i) {
    const int buf = i & 1, trow = (cta + i * NCTA) * ROWS;
    const uint32_t xu = su + F_X + buf * F_XB, xnu = su + F_XN + buf * F_XB;
#if CPX
    if (i == 0) {
      asm volatile("cp.async.wait_all;" ::: "memory");
      named_bar_sync(1 + wg, 128);                         // this warpgroup's 64 rows are in shared memory
    } else {
      mbar_wait(x_full + buf, ((i >> 1) & 1) ^ (buf == 0 ? 1 : 0));   // buffer 0's barrier completes its first phase at tile 2
    }
#else
    mbar_wait(x_full + buf, (i >> 1) & 1);
#endif
    TR(1);
#if GBSMEM
    if (i == 0) asm volatile("bar.sync 4, 256;\n" ::: "memory");   // gamma/beta staged by all threads
#endif
#if LNFRAG
    // ---------------------------------------------------------------- LayerNorm in the A-fragment layout (see transition_fwd_d64.cu): the x tile is read
    // with ldmatrix.x4 exactly as the wgmma A operand is laid out, the four lanes of a quad hold a row (two shuffles per reduction), and xn
    // stays in registers as the RS operand of the first GEMM of every chunk -- no shared-memory store, fence or barrier.
    uint32_t af[8][4];
    {
      const int mi = lane >> 3, mrow = wg * WGR + 16 * warp + 8 * (mi & 1) + (lane & 7);
#pragma unroll
      for (int ks = 0; ks < 8; ++ks) {
        const int col = 8 * (2 * ks + (mi >> 1));
        ldsm_x4(af[ks], xu + (col >> 6) * 16384 + swz128((uint32_t)mrow, (uint32_t)((col & 63) * 2)));
      }
    }
    {
      const int q = lane & 3;
      float s2[2], mean2[2], v2[2], rs2[2];
#pragma unroll
      for (int R = 0; R < 2; ++R) {
        float s_ = 0.f;
#pragma unroll
        for (int ks = 0; ks < 8; ++ks) s_ += (bf16lo(af[ks][R]) + bf16hi(af[ks][R])) + (bf16lo(af[ks][R + 2]) + bf16hi(af[ks][R + 2]));
        s2[R] = s_;
      }
#pragma unroll
      for (int o = 1; o <= 2; o <<= 1)
#pragma unroll
        for (int R = 0; R < 2; ++R) s2[R] += __shfl_xor_sync(0xffffffffu, s2[R], o);
#pragma unroll
      for (int R = 0; R < 2; ++R) {
        mean2[R] = s2[R] * (1.f / D_);
        float q_ = 0.f;
#pragma unroll
        for (int ks = 0; ks < 8; ++ks) {
          float d;
          d = bf16lo(af[ks][R]) - mean2[R]; q_ += d * d;
          d = bf16hi(af[ks][R]) - mean2[R]; q_ += d * d;
          d = bf16lo(af[ks][R + 2]) - mean2[R]; q_ += d * d;
          d = bf16hi(af[ks][R + 2]) - mean2[R]; q_ += d * d;
        }
        v2[R] = q_;
      }
#pragma unroll
      for (int o = 1; o <= 2; o <<= 1)
#pragma unroll
        for (int R = 0; R < 2; ++R) v2[R] += __shfl_xor_sync(0xffffffffu, v2[R], o);
#pragma unroll
      for (int R = 0; R < 2; ++R) rs2[R] = rsqrtf(v2[R] * (1.f / D_) + eps);
#pragma unroll
      for (int ks = 0; ks < 8; ++ks) {
        float gq[4], bq[4];                              // gamma / beta of columns 16 ks + 2 q + {0, 1} and + 8
#pragma unroll
        for (int h = 0; h < 2; ++h) {
#if GBSMEM
          const float2 gg = *reinterpret_cast<const float2*>(gbs + 16 * ks + 8 * h + 2 * q);
          const float2 bb = *reinterpret_cast<const float2*>(gbs + 128 + 16 * ks + 8 * h + 2 * q);
#else
          const float2 gg = *reinterpret_cast<const float2*>(p.gamma + 16 * ks + 8 * h + 2 * q);
          const float2 bb = *reinterpret_cast<const float2*>(p.beta + 16 * ks + 8 * h + 2 * q);
#endif
          gq[2 * h] = gg.x; gq[2 * h + 1] = gg.y; bq[2 * h] = bb.x; bq[2 * h + 1] = bb.y;
        }
#pragma unroll
        for (int R = 0; R < 2; ++R) {
          const uint32_t u0 = af[ks][R], u1 = af[ks][R + 2];
          af[ks][R] = pack_bf16((bf16lo(u0) - mean2[R]) * rs2[R] * gq[0] + bq[0], (bf16hi(u0) - mean2[R]) * rs2[R] * gq[1] + bq[1]);
          af[ks][R + 2] = pack_bf16((bf16lo(u1) - mean2[R]) * rs2[R] * gq[2] + bq[2], (bf16hi(u1) - mean2[R]) * rs2[R] * gq[3] + bq[3]);
        }
      }
    }
#if LNFRAG == 3 || LNFRAG == 4
    {   // experiment: same fragment LN, result written to the xn tile, SS wgmma afterwards
      const int g = lane >> 2, q = lane & 3;
#pragma unroll
      for (int ks = 0; ks < 8; ++ks)
#pragma unroll
        for (int r = 0; r < 4; ++r) {
          const int row = wg * WGR + 16 * warp + g + 8 * (r & 1), col = 16 * ks + 2 * q + 8 * (r >> 1);
          asm volatile("st.shared.b32 [%0], %1;" :: "r"(xnu + (uint32_t)(col >> 6) * 16384u + swz128((uint32_t)row, (uint32_t)((col & 63) * 2))), "r"(af[ks][r]) : "memory");
        }
    }
    fence_proxy_async();
    named_bar_sync(1 + wg, 128);
#if LNFRAG == 4
    {   // RS operand re-read from the stored tile with ldmatrix (as the SS path would see it)
      const int mi = lane >> 3, mrow = wg * WGR + 16 * warp + 8 * (mi & 1) + (lane & 7);
#pragma unroll
      for (int ks = 0; ks < 8; ++ks) {
        const int col = 8 * (2 * ks + (mi >> 1));
        ldsm_x4(af[ks], xnu + (col >> 6) * 16384 + swz128((uint32_t)mrow, (uint32_t)((col & 63) * 2)));
      }
    }
#endif
#endif
#else
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
#else
        // The inference build keeps these stores in the code and skips them at run time (save = 0). Removing them at compile
        // time lets ptxas schedule the kernel into 155 registers, and that schedule measured 7-8 % SLOWER than the training
        // build even though it writes 38 MB less at L384; behind the runtime guard it keeps the training build's schedule and
        // is 15-16 % faster than the compiled-out form (L384 138.3 -> 116.9 us, L768 516.3 -> 432.5 us, output bit-identical).
        if (save) {
          stg64u(p.xn + (size_t)(trow + r) * D_ + c0, o.x, o.y);
          if (lane == 0) { const int gr = trow + r; p.rstd[gr] = rs; p.c1[gr] = mean * rs; }
        }
#endif
      }
    }
    fence_proxy_async();                                  // the generic stores of xn -> visible to the wgmma operand reads
    named_bar_sync(1 + wg, 128);
#endif
    TR(2);
#if ABL == 1
    if (tid >= 0) { mbar_wait(w_full, 0); mbar_wait(w_full + 1, 0); return; }
#endif
    // ---------------------------------------------------------------- the eight hidden chunks
    float acc[64];
#pragma unroll
    for (int e = 0; e < 64; ++e) acc[e] = 0.f;
#if SUBCH
    {
      static_assert(LNFRAG == 1 && FWD_SAVE == 0 && !RESIDENT, "SUBCH is the fragment-LN inference build");
      constexpr int T = 2 * NCH;                                    // 32-unit sub-chunks
      float AB[2][32];
      uint32_t fhs[2][2][4];
      auto issue_g1 = [&](int t) {
        const int j = t >> 1, sc = t & 1;
        const int sl = (int)((wseq + j) % NSLOT);
        const uint32_t slot = su + F_RING + sl * F_SLOT;
        if (sc == 0) mbar_wait(w_full + sl, (uint32_t)(((wseq + j) / NSLOT) & 1));
#pragma unroll
        for (int e = 0; e < 32; ++e) AB[t & 1][e] = 0.f;
        fence_regs(AB[t & 1]); fence_regs(acc); wgmma_fence();
#pragma unroll
        for (int ks = 0; ks < 8; ++ks) {
          const uint64_t bdk = dsc_(slot + sc * 8192 + (ks >> 2) * 16384 + (ks & 3) * 32, 16, 1024);
          mma64_rs_k(AB[t & 1], af[ks], (uint32_t)bdk, (uint32_t)(bdk >> 32), ks > 0);
        }
        wgmma_commit();
      };
      issue_g1(0);
#pragma unroll
      for (int t = 0; t < T; ++t) {
        const int j = t >> 1, sc = t & 1;
        if (t + 1 < T) issue_g1(t + 1);
        if (t == 0 && t + 1 < T) wgmma_wait<1>();
        else if (t + 1 < T) wgmma_wait<2>();
        else wgmma_wait<1>();
        fence_regs(AB[t & 1]); fence_regs(acc);
        TR(3 + t);
        if ((t & 1) && t >= 3) {                                     // chunk (t - 3) / 2: both its sub-chunks' G2 have retired
          const uint32_t sq = wseq + (uint32_t)((t - 3) >> 1);
          const int ps = (int)(sq % NSLOT);
          if (wtid == 0) mbar_arrive(w_free + ps);
          if (tid == 0) { mbar_wait(w_free + ps, (uint32_t)((sq / NSLOT) & 1)); if (sq + NSLOT < wmax) issue_w(sq + NSLOT); }
        }
#pragma unroll
        for (int g = 0; g < 4; ++g) {
          const int e0 = 4 * g, ks = g >> 1, o = (g & 1) ? 2 : 0;
#pragma unroll
          for (int r = 0; r < 2; ++r) {
            const float a0 = AB[t & 1][e0 + 2 * r], a1 = AB[t & 1][e0 + 2 * r + 1];
            const float b0 = AB[t & 1][e0 + 16 + 2 * r], b1 = AB[t & 1][e0 + 16 + 2 * r + 1];
            fhs[t & 1][ks][o + r] = pack_bf16(a0 * sigmoid_kit(a0) * b0, a1 * sigmoid_kit(a1) * b1);
          }
        }
        fence_regs(acc); wgmma_fence();
        TR(13 + t);
        {
          const uint32_t slot = su + F_RING + (uint32_t)((wseq + j) % NSLOT) * F_SLOT;
          const uint64_t bd = dsc_(slot + F_WST, 8192, 1024);       // Ws^T_j is [K = 64 hs][N = 128 d]; this sub-chunk is rows 32 sc ..
          const uint32_t blo = (uint32_t)bd, bhi = (uint32_t)(bd >> 32);
#pragma unroll
          for (int ks = 0; ks < 2; ++ks) mma128_rs(acc, fhs[t & 1][ks], blo, bhi, (uint32_t)(sc * 4096 + ks * 2048) >> 4, 1);
        }
        wgmma_commit();
      }
      wgmma_wait<0>();
      {                                                             // the last chunk's slot (its G2 has retired)
        const uint32_t sq = wseq + NCH - 1;
        const int ps = (int)(sq % NSLOT);
        if (wtid == 0) mbar_arrive(w_free + ps);
        if (tid == 0) { mbar_wait(w_free + ps, (uint32_t)((sq / NSLOT) & 1)); if (sq + NSLOT < wmax) issue_w(sq + NSLOT); }
      }
      wseq += NCH;
    }
#else
#if (RSMASK >> 8) & 1
#pragma unroll                                            // experiment: no loop-carried RS operand across the back-edge
#endif
    for (int j = 0; j < (ABL == 3 ? 0 : NCH); ++j, ++wseq) {
      const int s = (int)(wseq % NSLOT);
      const uint32_t slot = su + F_RING + s * F_SLOT;
      mbar_wait(w_full + s, (uint32_t)((wseq / NSLOT) & 1));
      TR(3 + 4 * j);
      float AB[64];
#pragma unroll
      for (int e = 0; e < 64; ++e) AB[e] = 0.f;
      fence_regs(AB); fence_regs(acc);
#if SKEW
      if (pp && j == 0 && wg == 1) asm volatile("bar.sync 5, 256;\n" ::: "memory");
#endif
#if LNFRAG == 2
#pragma unroll
      for (int ks = 0; ks < 8; ++ks) fence_regs(af[ks]);
#endif
      wgmma_fence();
#pragma unroll
      for (int ks = 0; ks < 8; ++ks) {
#if LNFRAG == 1 || LNFRAG == 2 || LNFRAG == 4
        const uint64_t bdk = dk128(slot, ks);                  // B K-major: slab (ks >> 2) of the packed weight tile, + 32 B per k16 step
#if LNFRAG == 4
        if (((RSMASK & 0xFF) >> ks) & 1) mma128_rs_k(AB, af[ks], (uint32_t)bdk, (uint32_t)(bdk >> 32), 0u, ks > 0);
        else mma128<0, 0>(AB, dk128(xnu + wg * 8192, ks), bdk, ks > 0);
#else
        mma128_rs_k(AB, af[ks], (uint32_t)bdk, (uint32_t)(bdk >> 32), 0u, ks > 0);
#endif
#else
        mma128<0, 0>(AB, dk128(xnu + wg * 8192, ks), dk128(slot, ks), ks > 0);
#endif
      }
      wgmma_commit();
#if SKEW
      if (pp && j == 0 && wg == 0) asm volatile("bar.arrive 5, 256;\n" ::: "memory");
#endif
      wgmma_wait<1>(); fence_regs(acc);                   // also retires the previous chunk's squeeze update
#if !RESIDENT
      if (j > 0 || i > 0) {
        const int ps = (int)((wseq - 1) % NSLOT);
        if (pp) {
          if (wtid == 0) {                                // the second warpgroup to retire the slot refills it; nobody blocks
            uint32_t old;
            asm volatile("atom.add.acq_rel.cta.shared::cta.u32 %0, [%1], 1;" : "=r"(old) : "r"(smem_u32(w_cnt + ps)) : "memory");
            if (old == 1) {
              asm volatile("st.shared::cta.u32 [%0], 0;" :: "r"(smem_u32(w_cnt + ps)) : "memory");
              if (wseq + NSLOT - 1 < wmax) issue_w(wseq + NSLOT - 1);
            }
          }
        } else {
          if (wtid == 0) mbar_arrive(w_free + ps);
          if (tid == 0) { mbar_wait(w_free + ps, (uint32_t)(((wseq - 1) / NSLOT) & 1)); if (wseq + NSLOT - 1 < wmax) issue_w(wseq + NSLOT - 1); }
        }
      }
#endif
      wgmma_wait<0>(); fence_regs(AB);
      TR(4 + 4 * j);
#if LNFRAG == 2
#pragma unroll
      for (int ks = 0; ks < 8; ++ks) fence_regs(af[ks]);
#endif
      uint32_t fh[4][4];                                  // h = bf16(silu(a) b): C group g holds a, group g + 8 holds b
#pragma unroll
      for (int g = 0; g < 8; ++g) {
        const int e0 = 4 * g, ks = g >> 1, o = (g & 1) ? 2 : 0;
#pragma unroll
        for (int r = 0; r < 2; ++r) {
          const float a0 = AB[e0 + 2 * r], a1 = AB[e0 + 2 * r + 1];
          const float b0 = AB[e0 + 32 + 2 * r], b1 = AB[e0 + 32 + 2 * r + 1];
          fh[ks][o + r] = pack_bf16(a0 * sigmoid_kit(a0) * b0, a1 * sigmoid_kit(a1) * b1);
        }
      }
      fence_regs(acc); wgmma_fence();
      TR(5 + 4 * j);
      {
        const uint64_t bd = dsc_(slot + F_WST, 8192, 1024);   // Ws^T_j is [K = 64 hs][N = 128 d]
        const uint32_t blo = (uint32_t)bd, bhi = (uint32_t)(bd >> 32);
#pragma unroll
        for (int ks = 0; ks < 4; ++ks) mma128_rs(acc, fh[ks], blo, bhi, (uint32_t)(ks * 2048) >> 4, 1);
      }
      wgmma_commit();
    }

#endif
    wgmma_wait<0>(); fence_regs(acc);
    TR(30);
#if ABL == 2
    if (tid >= 0) return;
#endif
    // ---------------------------------------------------------------- out = bf16(x + acc), added IN PLACE over the x tile and
    // handed to one TMA store per warpgroup half.  Straight 4-byte global stores from this fragment layout touch eight
    // half-used 32-byte sectors per instruction and measured 60 us of the kernel's 145.
    // ldmatrix.x4 hands each thread its x values in exactly the accumulator's fragment layout (row lane / 4, columns
    // 2 (lane % 4) + {0, 1} of each 8 x 8 matrix), so four matrices per instruction replace sixteen 4-byte accesses; the add
    // is the base kernel's, element for element.  Matrices per x4: (rows 0-7, g), (rows 8-15, g), (0-7, g + 1), (8-15, g + 1).
    {
      const int mi = lane >> 3, mrow = wg * WGR + 16 * warp + 8 * (mi & 1) + (lane & 7);
#pragma unroll
      for (int gp = 0; gp < 8; ++gp) {
        const int col = 8 * (2 * gp + (mi >> 1));
        const uint32_t ad = xu + (col >> 6) * 16384 + swz128((uint32_t)mrow, (uint32_t)((col & 63) * 2));
        uint32_t xr[4];
        ldsm_x4(xr, ad);
        const int g0 = 2 * gp, g1 = 2 * gp + 1;
        stsm_x4(ad, pack_bf16(bf16lo(xr[0]) + acc[4 * g0 + 0], bf16hi(xr[0]) + acc[4 * g0 + 1]),
                    pack_bf16(bf16lo(xr[1]) + acc[4 * g0 + 2], bf16hi(xr[1]) + acc[4 * g0 + 3]),
                    pack_bf16(bf16lo(xr[2]) + acc[4 * g1 + 0], bf16hi(xr[2]) + acc[4 * g1 + 1]),
                    pack_bf16(bf16lo(xr[3]) + acc[4 * g1 + 2], bf16hi(xr[3]) + acc[4 * g1 + 3]));
      }
    }
    fence_proxy_async();
    named_bar_sync(1 + wg, 128);
    if (wtid == 0) {
#pragma unroll
      for (int c = 0; c < 2; ++c) tma_store_2d(p.outm, sm + F_X + buf * F_XB + c * 16384 + wg * 8192, c * 64, trow + wg * WGR);
      tma_store_commit();
      tma_store_wait_all();                                // the store has read the tile: the buffer may be refilled
      TR(31);
      mbar_arrive(x_free + buf);
    }
    if (tid == 0 && i + 2 < n_local) { mbar_wait(x_free + buf, (i >> 1) & 1); issue_x(i + 2); }
  }
}

// ================================================================================== SPLIT2: 64-row tile, hidden split over the two warpgroups
// Latency variant for few rows (one tile per CTA, n = 2: four hidden chunks). The 128-row tile keeps both warpgroups on
// different rows and every chunk on both, so a CTA holds 128 rows' worth of MMA (3.4 us) for as few as 24 CTAs. Here a CTA
// owns 64 rows, warpgroup w runs chunks w and w + 2 (resident, never recycled; chunks 0/1 are fetched first so the pair
// starts at once), and the two partial y accumulators are reduced through shared memory: each warpgroup hands the other the
// half of the columns it finalises (warpgroup w: columns 64 w .. 64 w + 63, the x slab w).
#if SPLIT2
static_assert(MW_N == 2 && LNFRAG == 1, "SPLIT2 is the n = 2 inference build with the fragment LayerNorm");
constexpr int S_SLOT = 49152, S_X = 4 * S_SLOT, S_BAR = S_X + 16384;
static_assert(S_BAR + 128 <= SMEM_BYTES, "shared memory budget");
extern "C" __global__ void __launch_bounds__(256, 1)
transition_fwd_fused_split2(const __grid_constant__ CUtensorMap mx, const __grid_constant__ CUtensorMap mwa,
                            const __grid_constant__ CUtensorMap mwb, const __grid_constant__ CUtensorMap mwst,
                            const __grid_constant__ CUtensorMap mout, const float* __restrict__ gamma,
                            const float* __restrict__ beta, __nv_bfloat16* __restrict__ xn, __nv_bfloat16* __restrict__ out,
                            float* __restrict__ rstd, float* __restrict__ c1, int M, int tiles, float eps, int save,
                     const __nv_bfloat16* __restrict__ xin) {
  (void)xn; (void)out; (void)rstd; (void)c1; (void)M; (void)save; (void)xin;
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  const int tid = threadIdx.x, wg = tid >> 7, wtid = tid & 127, warp = wtid >> 5, lane = tid & 31;
  const int cta = blockIdx.x;
  if (cta >= tiles) return;
  uint64_t* bars = reinterpret_cast<uint64_t*>(sm + S_BAR);        // w_full[4], x_full
  const int row0 = cta * 64;
  if (tid == 0) {
    tma_prefetch_desc(&mx); tma_prefetch_desc(&mwa); tma_prefetch_desc(&mwb); tma_prefetch_desc(&mwst); tma_prefetch_desc(&mout);
    for (int k = 0; k < 5; ++k) mbar_init(bars + k, 1);
    fence_barrier_init();
  }
  __syncthreads();
  if (tid == 0) {
    mbar_arrive_expect_tx(bars + 4, 16384);
#pragma unroll
    for (int c = 0; c < 2; ++c) tma_load_2d(sm + S_X + c * 8192, &mx, bars + 4, c * 64, row0);
#pragma unroll
    for (int jc = 0; jc < 4; ++jc) {                               // order 0, 1, 2, 3: the first two feed both warpgroups
      uint8_t* b = sm + jc * S_SLOT;
      mbar_arrive_expect_tx(bars + jc, 49152);
#pragma unroll
      for (int c = 0; c < 2; ++c) {
        tma_load_2d(b + c * 16384, &mwa, bars + jc, c * 64, jc * HS);
        tma_load_2d(b + c * 16384 + 8192, &mwb, bars + jc, c * 64, jc * HS);
        tma_load_2d(b + F_WST + c * 8192, &mwst, bars + jc, c * 64, jc * HS);
      }
    }
  }
  const uint32_t xs = su + S_X;
  uint32_t af[8][4];
  mbar_wait(bars + 4, 0);
  {
    const int mi = lane >> 3, mrow = 16 * warp + 8 * (mi & 1) + (lane & 7);
#pragma unroll
    for (int ks = 0; ks < 8; ++ks) {
      const int col = 8 * (2 * ks + (mi >> 1));
      ldsm_x4(af[ks], xs + (col >> 6) * 8192 + swz128((uint32_t)mrow, (uint32_t)((col & 63) * 2)));
    }
  }
  {
    const int q = lane & 3;
    float s2[2], mean2[2], v2[2], rs2[2];
#pragma unroll
    for (int R = 0; R < 2; ++R) {
      float s_ = 0.f;
#pragma unroll
      for (int ks = 0; ks < 8; ++ks) s_ += (bf16lo(af[ks][R]) + bf16hi(af[ks][R])) + (bf16lo(af[ks][R + 2]) + bf16hi(af[ks][R + 2]));
      s2[R] = s_;
    }
#pragma unroll
    for (int o = 1; o <= 2; o <<= 1)
#pragma unroll
      for (int R = 0; R < 2; ++R) s2[R] += __shfl_xor_sync(0xffffffffu, s2[R], o);
#pragma unroll
    for (int R = 0; R < 2; ++R) {
      mean2[R] = s2[R] * (1.f / D_);
      float q_ = 0.f;
#pragma unroll
      for (int ks = 0; ks < 8; ++ks) {
        float d;
        d = bf16lo(af[ks][R]) - mean2[R]; q_ += d * d;
        d = bf16hi(af[ks][R]) - mean2[R]; q_ += d * d;
        d = bf16lo(af[ks][R + 2]) - mean2[R]; q_ += d * d;
        d = bf16hi(af[ks][R + 2]) - mean2[R]; q_ += d * d;
      }
      v2[R] = q_;
    }
#pragma unroll
    for (int o = 1; o <= 2; o <<= 1)
#pragma unroll
      for (int R = 0; R < 2; ++R) v2[R] += __shfl_xor_sync(0xffffffffu, v2[R], o);
#pragma unroll
    for (int R = 0; R < 2; ++R) rs2[R] = rsqrtf(v2[R] * (1.f / D_) + eps);
#pragma unroll
    for (int ks = 0; ks < 8; ++ks) {
      float gq[4], bq[4];
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const float2 gg = *reinterpret_cast<const float2*>(gamma + 16 * ks + 8 * h + 2 * q);
        const float2 bb = *reinterpret_cast<const float2*>(beta + 16 * ks + 8 * h + 2 * q);
        gq[2 * h] = gg.x; gq[2 * h + 1] = gg.y; bq[2 * h] = bb.x; bq[2 * h + 1] = bb.y;
      }
#pragma unroll
      for (int R = 0; R < 2; ++R) {
        const uint32_t u0 = af[ks][R], u1 = af[ks][R + 2];
        af[ks][R] = pack_bf16((bf16lo(u0) - mean2[R]) * rs2[R] * gq[0] + bq[0], (bf16hi(u0) - mean2[R]) * rs2[R] * gq[1] + bq[1]);
        af[ks][R + 2] = pack_bf16((bf16lo(u1) - mean2[R]) * rs2[R] * gq[2] + bq[2], (bf16hi(u1) - mean2[R]) * rs2[R] * gq[3] + bq[3]);
      }
    }
  }
  float acc[64];
#pragma unroll
  for (int e = 0; e < 64; ++e) acc[e] = 0.f;
#pragma unroll
  for (int jj = 0; jj < 2; ++jj) {
    const int j = wg + 2 * jj;
    const uint32_t slot = su + (uint32_t)j * S_SLOT;
    mbar_wait(bars + j, 0);
    float AB[64];
#pragma unroll
    for (int e = 0; e < 64; ++e) AB[e] = 0.f;
    fence_regs(AB); fence_regs(acc); wgmma_fence();
#pragma unroll
    for (int ks = 0; ks < 8; ++ks) {
      const uint64_t bdk = dk128(slot, ks);
      mma128_rs_k(AB, af[ks], (uint32_t)bdk, (uint32_t)(bdk >> 32), 0u, ks > 0);
    }
    wgmma_commit();
    wgmma_wait<0>(); fence_regs(AB); fence_regs(acc);
    uint32_t fh[4][4];
#pragma unroll
    for (int g = 0; g < 8; ++g) {
      const int e0 = 4 * g, ks = g >> 1, o = (g & 1) ? 2 : 0;
#pragma unroll
      for (int r = 0; r < 2; ++r) {
        const float a0 = AB[e0 + 2 * r], a1 = AB[e0 + 2 * r + 1];
        const float b0 = AB[e0 + 32 + 2 * r], b1 = AB[e0 + 32 + 2 * r + 1];
        fh[ks][o + r] = pack_bf16(a0 * sigmoid_kit(a0) * b0, a1 * sigmoid_kit(a1) * b1);
      }
    }
    fence_regs(acc); wgmma_fence();
    {
      const uint64_t bd = dsc_(slot + F_WST, 8192, 1024);
      const uint32_t blo = (uint32_t)bd, bhi = (uint32_t)(bd >> 32);
#pragma unroll
      for (int ks = 0; ks < 4; ++ks) mma128_rs(acc, fh[ks], blo, bhi, (uint32_t)(ks * 2048) >> 4, 1);
    }
    wgmma_commit();
  }
  wgmma_wait<0>(); fence_regs(acc);
  // ---- reduce: warpgroup w keeps columns 64 w.., so it hands the other warpgroup acc[32 (1 - w) .. 32 (1 - w) + 31]
  named_bar_sync(3, 256);                                          // every MMA of both warpgroups is done: the weight slots are free
  {
    const uint32_t st = su + (uint32_t)wg * 16384u;                // stage[wg]: [8 e4][128 threads] float4
#pragma unroll
    for (int e4 = 0; e4 < 8; ++e4) {
      const int e = 32 * (1 - wg) + 4 * e4;
      asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(st + (uint32_t)(e4 * 128 + wtid) * 16u),
                   "f"(acc[e]), "f"(acc[e + 1]), "f"(acc[e + 2]), "f"(acc[e + 3]) : "memory");
    }
  }
  named_bar_sync(3, 256);
  {
    const uint32_t ld = su + (uint32_t)(1 - wg) * 16384u;          // what the other warpgroup staged for this one
#pragma unroll
    for (int e4 = 0; e4 < 8; ++e4) {
      float4 v;
      asm volatile("ld.shared.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w)
                   : "r"(ld + (uint32_t)(e4 * 128 + wtid) * 16u) : "memory");
      const int e = 32 * wg + 4 * e4;
      acc[e] += v.x; acc[e + 1] += v.y; acc[e + 2] += v.z; acc[e + 3] += v.w;
    }
  }
  {
    const int mi = lane >> 3, mrow = 16 * warp + 8 * (mi & 1) + (lane & 7);
#pragma unroll
    for (int gq = 0; gq < 4; ++gq) {
      const int gp = 4 * wg + gq;
      const int col = 8 * (2 * gp + (mi >> 1));
      const uint32_t ad = xs + (col >> 6) * 8192 + swz128((uint32_t)mrow, (uint32_t)((col & 63) * 2));
      uint32_t xr[4];
      ldsm_x4(xr, ad);
      const int g0 = 2 * gp, g1 = 2 * gp + 1;
      stsm_x4(ad, pack_bf16(bf16lo(xr[0]) + acc[4 * g0 + 0], bf16hi(xr[0]) + acc[4 * g0 + 1]),
                  pack_bf16(bf16lo(xr[1]) + acc[4 * g0 + 2], bf16hi(xr[1]) + acc[4 * g0 + 3]),
                  pack_bf16(bf16lo(xr[2]) + acc[4 * g1 + 0], bf16hi(xr[2]) + acc[4 * g1 + 1]),
                  pack_bf16(bf16lo(xr[3]) + acc[4 * g1 + 2], bf16hi(xr[3]) + acc[4 * g1 + 3]));
    }
  }
  fence_proxy_async();
  named_bar_sync(1 + wg, 128);
  if (wtid == 0) {
    tma_store_2d(&mout, sm + S_X + wg * 8192, wg * 64, row0);
    tma_store_commit();
    tma_store_wait_all();
  }
}
#endif

// ================================================================================== host launcher
// Wiring surface for `fused_sm90a.py`. The kernel is a persistent grid: one CTA per SM, NCTA
// baked in at build time from the device's own multiprocessor count, so there is nothing to
// choose at launch. The opt-in shared-memory attribute is set once per process: sm_90's
// effective dynamic ceiling is 231424 B, not the 232448 B the occupancy API advertises.

#include <stdexcept>
#include <string>

#if TRACE
void transition_fused_trace(unsigned long long* host) { cudaDeviceSynchronize(); cudaMemcpyFromSymbol(host, g_trace, sizeof(unsigned long long) * 64); }
#else
void transition_fused_trace(unsigned long long* host) { for (int i = 0; i < 64; ++i) host[i] = 0; }
#endif
int transition_fused_fwd_ctas() { return NCTA; }
int transition_fused_fwd_rows() { return SPLIT2 ? 64 : ROWS; }
bool transition_fused_fwd_saves_xn() { return FWD_SAVE != 0; }

void transition_fused_fwd_launch(
    const CUtensorMap& mx, const CUtensorMap& mwa, const CUtensorMap& mwb,
    const CUtensorMap& mwst, const CUtensorMap& mout,
    const float* gamma, const float* beta, __nv_bfloat16* xn, __nv_bfloat16* out,
    float* rstd, float* c1, int M, int tiles, float eps, const __nv_bfloat16* xin, cudaStream_t stream) {
  const int save = FWD_SAVE != 0;                        // the inference build (FWD_SAVE = 0) is always launched with save = 0
#if SPLIT2
  {
  static const bool ready = [] {
    cudaError_t e = cudaFuncSetAttribute(reinterpret_cast<const void*>(transition_fwd_fused_split2),
                                         cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES);
    if (e != cudaSuccess)
      throw std::runtime_error(std::string("transition_fwd_fused_split2 smem opt-in: ") + cudaGetErrorString(e));
    return true;
  }();
  (void)ready;
  transition_fwd_fused_split2<<<(tiles < NCTA ? tiles : NCTA), 256, SMEM_BYTES, stream>>>(
      mx, mwa, mwb, mwst, mout, gamma, beta, xn, out, rstd, c1, M, tiles, eps, save, xin);
  return;
  }
#endif
  static const bool ready = [] {
    cudaError_t e = cudaFuncSetAttribute(reinterpret_cast<const void*>(transition_fwd_fused),
                                         cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES);
    if (e != cudaSuccess)
      throw std::runtime_error(std::string("transition_fwd_fused smem opt-in: ") + cudaGetErrorString(e));
    return true;
  }();
  (void)ready;
#if PDL
  {
    cudaLaunchConfig_t cfg{};
    cfg.gridDim = dim3((unsigned)(tiles < NCTA ? tiles : NCTA), 1, 1);
    cfg.blockDim = dim3(256, 1, 1);
    cfg.dynamicSmemBytes = SMEM_BYTES;
    cfg.stream = stream;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attr; cfg.numAttrs = 1;
    cudaError_t e = cudaLaunchKernelEx(&cfg, transition_fwd_fused, mx, mwa, mwb, mwst, mout, gamma, beta, xn, out, rstd, c1, M, tiles, eps, save, xin);
    if (e != cudaSuccess) throw std::runtime_error(std::string("transition_fwd_fused launch: ") + cudaGetErrorString(e));
  }
#else
  transition_fwd_fused<<<(tiles < NCTA ? tiles : NCTA), 256, SMEM_BYTES, stream>>>(   // only the CTAs that have a tile
      mx, mwa, mwb, mwst, mout, gamma, beta, xn, out, rstd, c1, M, tiles, eps, save, xin);
#endif
}
