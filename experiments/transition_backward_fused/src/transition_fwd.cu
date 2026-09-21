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
constexpr int D_ = 128, H_ = 512, HS = 64, NCH = H_ / HS, ROWS = 128, WGR = 64;

// ---- shared memory: a two-slot weight ring plus double-buffered x and xn tiles
constexpr int F_SLOT = 49152, F_RING = 0;                      // slot: [Wa_j; Wb_j] 32 KB at +0, then Ws^T_j 16 KB at +32768
constexpr int F_WST = 32768;
constexpr int F_X = 2 * F_SLOT, F_XB = 32768;                  // x  tile [128 rows][128 d] bf16, double-buffered
constexpr int F_XN = F_X + 2 * F_XB;                           // xn tile, double-buffered
constexpr int F_BAR = F_XN + 2 * F_XB;
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


struct Par {                                 // the tensor maps stay in the grid-constant parameter bank
  const CUtensorMap *x, *wa, *wb, *wst;
  const float *gamma, *beta;
  __nv_bfloat16 *xn, *out;
  float *rstd, *c1;
  int M, tiles;
};

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
                     const float* __restrict__ gamma, const float* __restrict__ beta,
                     __nv_bfloat16* __restrict__ xn, __nv_bfloat16* __restrict__ out,
                     float* __restrict__ rstd, float* __restrict__ c1, int M, int tiles, float eps) {
  const Par p{&mx, &mwa, &mwb, &mwst, gamma, beta, xn, out, rstd, c1, M, tiles};
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  const int tid = threadIdx.x, wg = tid >> 7, wtid = tid & 127, warp = wtid >> 5, lane = tid & 31;
  const int cta = blockIdx.x;
  uint64_t* bars = reinterpret_cast<uint64_t*>(sm + F_BAR);
  uint64_t* w_full = bars; uint64_t* w_free = bars + 2; uint64_t* x_full = bars + 4; uint64_t* x_free = bars + 6;
  const int n_local = (tiles > cta) ? (tiles - cta + NCTA - 1) / NCTA : 0;
  const uint32_t wmax = (uint32_t)n_local * NCH;
  auto issue_w = [&](uint32_t seq) {                      // chunk seq % NCH into slot seq & 1
    const int s = seq & 1, j = (int)(seq % NCH); uint8_t* b = sm + F_RING + s * F_SLOT;
    mbar_arrive_expect_tx(w_full + s, 49152);
#pragma unroll
    for (int c = 0; c < 2; ++c) {
      tma_load_2d(b + c * 16384, p.wa, w_full + s, c * 64, j * HS);            // rows 0..63 of the packed [Wa_j; Wb_j] tile
      tma_load_2d(b + c * 16384 + 8192, p.wb, w_full + s, c * 64, j * HS);      // rows 64..127
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
  if (tid == 0) {
    mbar_init(w_full, 1); mbar_init(w_full + 1, 1); mbar_init(w_free, 2); mbar_init(w_free + 1, 2);
    mbar_init(x_full, 1); mbar_init(x_full + 1, 1); mbar_init(x_free, 2); mbar_init(x_free + 1, 2);
    fence_barrier_init();
  }
  __syncthreads();
  uint32_t wseq = 0;
  if (tid == 0) { if (wmax > 0) issue_w(0); if (wmax > 1) issue_w(1); for (int i = 0; i < 2 && i < n_local; ++i) issue_x(i); }
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
    mbar_wait(x_full + buf, (i >> 1) & 1);
    // ---------------------------------------------------------------- LayerNorm, in registers, into the xn tile
    for (int ii = 0; ii < 16; ++ii) {
      const int r = wg * WGR + 16 * warp + ii;
      const uint32_t off = (uint32_t)r * 128u + ((gran ^ ((uint32_t)r & 7u)) << 4);
      const uint2 v = lds64u(xu + xcol + off);
      const float x0 = bf16lo(v.x), x1 = bf16hi(v.x), x2 = bf16lo(v.y), x3 = bf16hi(v.y);
      const float mean = warp_allsum(((x0 + x1) + (x2 + x3))) * (1.f / D_);
      float d, q = 0.f;
      d = x0 - mean; q += d * d; d = x1 - mean; q += d * d; d = x2 - mean; q += d * d; d = x3 - mean; q += d * d;
      const float rs = rsqrtf(warp_allsum(q) * (1.f / D_) + eps);
      uint2 o;
      o.x = pack_bf16((x0 - mean) * rs * g4[0] + b4[0], (x1 - mean) * rs * g4[1] + b4[1]);
      o.y = pack_bf16((x2 - mean) * rs * g4[2] + b4[2], (x3 - mean) * rs * g4[3] + b4[3]);
      asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(xnu + xcol + off), "r"(o.x), "r"(o.y) : "memory");
      stg64u(p.xn + (size_t)(trow + r) * D_ + c0, o.x, o.y);   // the backward's saved xn: 32 lanes x 8 B = one whole row
      if (lane == 0) { const int gr = trow + r; p.rstd[gr] = rs; p.c1[gr] = mean * rs; }
    }
    fence_proxy_async();                                  // the generic stores of xn -> visible to the wgmma operand reads
    named_bar_sync(1 + wg, 128);
    // ---------------------------------------------------------------- the eight hidden chunks
    float acc[64];
#pragma unroll
    for (int e = 0; e < 64; ++e) acc[e] = 0.f;
    for (int j = 0; j < NCH; ++j, ++wseq) {
      const int s = (int)(wseq & 1);
      const uint32_t slot = su + F_RING + s * F_SLOT;
      mbar_wait(w_full + s, (uint32_t)((wseq >> 1) & 1));
      float AB[64];
#pragma unroll
      for (int e = 0; e < 64; ++e) AB[e] = 0.f;
      fence_regs(AB); fence_regs(acc); wgmma_fence();
#pragma unroll
      for (int ks = 0; ks < 8; ++ks) mma128<0, 0>(AB, dk128(xnu + wg * 8192, ks), dk128(slot, ks), ks > 0);
      wgmma_commit();
      wgmma_wait<1>(); fence_regs(acc);                   // also retires the previous chunk's squeeze update
      if (j > 0 || i > 0) {
        const int ps = (int)((wseq - 1) & 1);
        if (wtid == 0) mbar_arrive(w_free + ps);
        if (tid == 0) { mbar_wait(w_free + ps, (uint32_t)(((wseq - 1) >> 1) & 1)); if (wseq + 1 < wmax) issue_w(wseq + 1); }
      }
      wgmma_wait<0>(); fence_regs(AB);
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
      {
        const uint64_t bd = dsc_(slot + F_WST, 8192, 1024);   // Ws^T_j is [K = 64 hs][N = 128 d]
        const uint32_t blo = (uint32_t)bd, bhi = (uint32_t)(bd >> 32);
#pragma unroll
        for (int ks = 0; ks < 4; ++ks) mma128_rs(acc, fh[ks], blo, bhi, (uint32_t)(ks * 2048) >> 4, 1);
      }
      wgmma_commit();
    }
    wgmma_wait<0>(); fence_regs(acc);
    // ---------------------------------------------------------------- out = bf16(x + acc), in the m64n128 fragment layout
    const int lrow = wg * WGR + 16 * warp + (lane >> 2);
#pragma unroll
    for (int rb = 0; rb < 2; ++rb) {
      const int r = lrow + 8 * rb, grow = trow + r;
#pragma unroll
      for (int g = 0; g < 16; ++g) {
        const int col = 8 * g + 2 * (lane & 3);
        const uint32_t xv = lds32(xu + (col >> 6) * 16384 + swz128((uint32_t)r, (uint32_t)((col & 63) * 2)));
        stg32u(p.out + (size_t)grow * D_ + col,
               pack_bf16(bf16lo(xv) + acc[4 * g + 2 * rb], bf16hi(xv) + acc[4 * g + 2 * rb + 1]));
      }
    }
    if (wtid == 0) mbar_arrive(x_free + buf);
    if (tid == 0 && i + 2 < n_local) { mbar_wait(x_free + buf, (i >> 1) & 1); issue_x(i + 2); }
  }
}
