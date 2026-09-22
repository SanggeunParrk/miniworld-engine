// p19: LayerNorm with two threads per row, the base kernel's reduction tree reproduced (bit-identical).
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
#define FWD_SAVE 1          // 1: also write xn, rstd and c1 (what the backward needs). 0: inference, output only.
#endif
constexpr int D_ = 128, H_ = 512, HS = 64, NCH = H_ / HS, ROWS = 128, WGR = 64;

// ---- shared memory: a two-slot weight ring plus double-buffered x and xn tiles
constexpr int F_SLOT = 49152, F_RING = 0;                      // slot: [Wa_j; Wb_j] 32 KB at +0, then Ws^T_j 16 KB at +32768
constexpr int F_WST = 32768;
constexpr int F_X = 2 * F_SLOT, F_XB = 32768;                  // x  tile [128 rows][128 d] bf16, double-buffered
constexpr int F_XN = F_X + 2 * F_XB;                           // xn tile, double-buffered
constexpr int F_BAR = F_XN + 2 * F_XB;
constexpr int F_GB = F_BAR + 256;                            // gamma [128] | beta [128] fp32
constexpr int SMEM_BYTES = 231424;
static_assert(F_GB + 1024 <= SMEM_BYTES, "shared memory budget");

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
  {                                                       // gamma | beta, fp32, 1 KB: every thread of a half reads the same granule (broadcast)
    float* gb = reinterpret_cast<float*>(sm + F_GB);
    if (tid < 128) gb[tid] = p.gamma[tid]; else gb[tid] = p.beta[tid - 128];
  }
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

  for (int i = 0; i < n_local; ++i) {
    const int buf = i & 1, trow = (cta + i * NCTA) * ROWS;
    const uint32_t xu = su + F_X + buf * F_XB, xnu = su + F_XN + buf * F_XB;
    mbar_wait(x_full + buf, (i >> 1) & 1);
    // ---------------------------------------------------------------- LayerNorm, in registers, into the xn tile
    // Eight rows at a time: the two warp reductions a row needs are five dependent shuffles each, and one row at a time
    // left that latency fully exposed (34 us of 145 by ablation). Eight independent chains per step hide it; the reduction
    // order within a row is unchanged, so the statistics are bit-identical to the serial form.
    {
      // Two threads per row, each holding one K-half (64 columns) as 8 x 16 B granules.  The reduction reproduces the base
      // kernel's tree exactly: there, lane L summed its 4 columns and the warp butterflied xor 16, 8, 4, 2, 1.  Here "lane"
      // l = 0..15 of each half is a register; the xor-16 step pairs the two halves -- one shuffle with the partner thread,
      // then p_l + q_l (fp add is commutative, so both threads get the base kernel's bits) -- and xor 8, 4, 2, 1 run in
      // registers in the same order.  32 shuffles per warp per tile instead of 160, and 16-byte shared accesses.
      const int hh = wtid & 1, r = wg * WGR + (wtid >> 1);
      const uint32_t rbase = xu + (uint32_t)hh * 16384u + (uint32_t)r * 128u;
      uint4 xv[8];
#pragma unroll
      for (int g = 0; g < 8; ++g) xv[g] = lds128(rbase + (((uint32_t)g ^ ((uint32_t)r & 7u)) << 4));
      auto word = [&](int l, int w) __attribute__((always_inline)) -> uint32_t {        // local lane l's 4 columns: granule l >> 1, words 2 (l & 1) + w
        const uint4& q = xv[l >> 1];
        return (l & 1) ? (w ? q.w : q.z) : (w ? q.y : q.x);
      };
      auto tree = [&](float (&pt)[16]) __attribute__((always_inline)) -> float {        // the base kernel's butterfly, lane 0's value
#pragma unroll
        for (int l = 0; l < 16; ++l) pt[l] = pt[l] + __shfl_xor_sync(0xffffffffu, pt[l], 1);   // xor 16: the other half
#pragma unroll
        for (int l = 0; l < 8; ++l) pt[l] = pt[l] + pt[l + 8];                              // xor 8
#pragma unroll
        for (int l = 0; l < 4; ++l) pt[l] = pt[l] + pt[l + 4];                              // xor 4
        pt[0] = pt[0] + pt[2]; pt[1] = pt[1] + pt[3];                                         // xor 2
        return pt[0] + pt[1];                                                                 // xor 1
      };
      float part[16];
#pragma unroll
      for (int l = 0; l < 16; ++l) {
        const uint32_t vx = word(l, 0), vy = word(l, 1);
        part[l] = (bf16lo(vx) + bf16hi(vx)) + (bf16lo(vy) + bf16hi(vy));
      }
      const float mean = tree(part) * (1.f / D_);
#pragma unroll
      for (int l = 0; l < 16; ++l) {
        const uint32_t vx = word(l, 0), vy = word(l, 1);
        float acc = 0.f, d;
        d = bf16lo(vx) - mean; acc += d * d;
        d = bf16hi(vx) - mean; acc += d * d;
        d = bf16lo(vy) - mean; acc += d * d;
        d = bf16hi(vy) - mean; acc += d * d;
        part[l] = acc;
      }
      const float rs = rsqrtf(tree(part) * (1.f / D_) + eps);
      const uint32_t nbase = xnu + (uint32_t)hh * 16384u + (uint32_t)r * 128u;
      const float* gs = reinterpret_cast<const float*>(sm + F_GB) + 64 * hh;       // gamma[64 hh ..], then beta at +128
#pragma unroll
      for (int g = 0; g < 8; ++g) {
        uint32_t o[4];
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          const int l = 2 * g + e;
          const float4 g4 = *reinterpret_cast<const float4*>(gs + 4 * l), b4 = *reinterpret_cast<const float4*>(gs + 128 + 4 * l);
          const uint32_t vx = word(l, 0), vy = word(l, 1);
          o[2 * e] = pack_bf16((bf16lo(vx) - mean) * rs * g4.x + b4.x, (bf16hi(vx) - mean) * rs * g4.y + b4.y);
          o[2 * e + 1] = pack_bf16((bf16lo(vy) - mean) * rs * g4.z + b4.z, (bf16hi(vy) - mean) * rs * g4.w + b4.w);
        }
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(nbase + (((uint32_t)g ^ ((uint32_t)r & 7u)) << 4)),
                     "r"(o[0]), "r"(o[1]), "r"(o[2]), "r"(o[3]) : "memory");
#if FWD_SAVE
        asm volatile("st.global.v4.b32 [%0], {%1,%2,%3,%4};" :: "l"(p.xn + (size_t)(trow + r) * D_ + 64 * hh + 8 * g),
                     "r"(o[0]), "r"(o[1]), "r"(o[2]), "r"(o[3]) : "memory");
#endif
      }
#if FWD_SAVE
      if (hh == 0) { p.rstd[trow + r] = rs; p.c1[trow + r] = mean * rs; }
#endif
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
