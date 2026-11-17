// transition_fp8_fwd_sm90a.cu -- OPT-IN FP8 (e4m3) inference forward of the D128 Transition (engine 3.0.0 line, docs/fp8.md).
// SPDX-License-Identifier: Apache-2.0
// The 2.5.12 inference build of transition_fused_fwd_sm90a_kernel.cu (LNFRAG = 1, XFIRST, DECOUPLE, SKEW, unrolled chunk loop) with both
// GEMMs on the e4m3 tensor-core path (wgmma m64n128k32 .e4m3.e4m3, fp32 accumulation); the algorithm is the same:
//
//   xn8  = e4m3(LN(x) * sx)                         LayerNorm in the A-fragment layout, quantized in registers (no bf16 rounding first)
//   per hidden chunk j (64 units):
//     [A|B] = xn8 [Wa8_j; Wb8_j]^T                   RS m64n128k32 x 4, fp32
//     h8_j  = e4m3(silu(A ca) B cb)                  ca = 1 / (sx s1), cb = ca sh: the C fragment IS the k32 A fragment after a K permutation
//     acc  += h8_j Ws8_j^T                           RS m64n128k32 x 2
//   out = bf16(x + acc co)                           co = 1 / (sh s2)
//
// Scales (all powers of two, device float[8] read at kernel start, so a captured graph follows an in-place rescale):
//   sx: from the LayerNorm bound |xn| <= sqrt(D - 1) max|gamma| + max|beta| (no calibration, cannot saturate);
//   s1, s2: per-tensor weight scales (exact amax of the packed weights);
//   sh: static, calibrated on the first eager call (HEADROOM), guarded: every CTA reports max |h sh| (pre-cast) by atomicMax to `peak`.
// K permutation (both GEMMs): an fp32 C / bf16 A fragment holds columns {2t, 2t+1, 8+2t, 9+2t} (+16) of a 32-wide block, the e4m3 k32 A
// fragment wants logical columns {4t..4t+3} (+16): the packed weights store K in that logical order (host side), the activations stay put.
#include "tmn_kernels.cuh"
using namespace tmn; using namespace tmn::sm90;

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <array>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>

#ifndef NCTA
#define NCTA 132
#endif
#ifndef MW_N
#define MW_N 4
#endif
#ifndef WSLOTS
#define WSLOTS 4   // weight-ring slots (24 KB each in e4m3)
#endif
#ifndef SIGT
#define SIGT 1     // 1: sigmoid as 0.5 tanh.approx(a / 2) + 0.5 (one MUFU op); 0: the kit's rcp(1 + ex2(-a log2 e)) (two)
#endif
#ifndef F8G1
#define F8G1 1     // 1: first GEMM e4m3 (xn quantized in registers); 0: the bf16 expand of 2.5.12 (bf16 xn fragments, bf16 [Wa_j; Wb_j])
#endif
#ifndef F8G2
#define F8G2 1     // 1: second GEMM e4m3 (h quantized); 0: first GEMM only e4m3, h bf16 and Ws^T bf16 (the bf16 RS squeeze)
#endif
#ifndef ABL
#define ABL 0      // timing ablation only (wrong results): 3 = no chunk loop (LN + store only), 4 = no swiglu math (GEMMs only)
#endif

constexpr int D_ = 128, H_ = MW_N * 128, HS = 64, NCH = H_ / HS, ROWS = 128, WGR = 64;
// slot: [Wa8_j; Wb8_j] 128 rows x 128 B (128B swizzle) 16 KB, then the second-GEMM operand:
//   F8G2: Ws8_j [128 d][64 hs] e4m3, 64B swizzle, 8 KB;  else Ws^T_j bf16 [64 hs][128 d] (two 64-col 128B-swizzled boxes), 16 KB
constexpr int F_W2 = F8G1 ? 16384 : 32768;
constexpr int F_SLOT = F_W2 + (F8G2 ? 8192 : 16384);
constexpr int W_BYTES = F_SLOT;
constexpr int NSLOT = WSLOTS;
constexpr int F_X = NSLOT * F_SLOT, F_XB = 32768;
constexpr int F_BAR = F_X + 2 * F_XB;
constexpr int SMEM_BYTES = F_BAR + 1024;
static_assert(SMEM_BYTES <= 231424, "shared memory budget");

TMN_DEVI uint64_t dsc_(uint32_t addr, uint32_t lbo, uint32_t sbo, uint32_t lay) { return smem_desc(addr, lbo, sbo, lay); }

// e4m3x2: `a` -> upper byte, `b` -> lower byte
TMN_DEVI uint32_t f8x2(float lo, float hi) {
  uint16_t r;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;\n" : "=h"(r) : "f"(hi), "f"(lo));
  return (uint32_t)r;
}
TMN_DEVI uint32_t f8x4(float v0, float v1, float v2, float v3) { return f8x2(v0, v1) | (f8x2(v2, v3) << 16); }

TMN_DEVI float sigm(float a) {
#if SIGT
  float t;
  asm("tanh.approx.f32 %0, %1;\n" : "=f"(t) : "f"(0.5f * a));
  return fmaf(0.5f, t, 0.5f);
#else
  return rcpf(__fadd_rn(1.f, ex2f(__fmul_rn(-1.4426950408889634f, a))));
#endif
}

// m64n128k32 e4m3 x e4m3 -> fp32, A from registers, B K-major descriptor
TMN_DEVI void mma128_f8(float (&d)[64], const uint32_t (&a)[4], uint64_t bd, int accumulate) {
  asm volatile("{\n .reg .pred p;\n setp.ne.b32 p, %69, 0;\n"
    "wgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, {%64,%65,%66,%67}, %68, p, 1, 1;\n}\n"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(bd), "r"(accumulate));
}
// m64n128k16 bf16, A from registers, B K-major (trans-b = 0): the 2.5.12 expand
TMN_DEVI void mma128_bf_k(float (&d)[64], const uint32_t (&a)[4], uint64_t bd, int accumulate) {
  asm volatile("{\n .reg .pred p;\n setp.ne.b32 p, %69, 0;\n"
    "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, {%64,%65,%66,%67}, %68, p, 1, 1, 0;\n}\n"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(bd), "r"(accumulate));
}
#if !F8G2
// m64n128k16 bf16, A from registers, B MN-major (trans-b = 1): the 2.5.12 squeeze
TMN_DEVI void mma128_bf_rs(float (&d)[64], const uint32_t (&a)[4], uint64_t bd) {
  asm volatile("{\n .reg .pred p;\n setp.ne.b32 p, 1, 0;\n"
    "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, {%64,%65,%66,%67}, %68, p, 1, 1, 1;\n}\n"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(bd));
}
#endif

TMN_DEVI void tma_store_2d(const CUtensorMap* map, const void* src, int c0, int c1) {
  asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%2, %3}], [%1];"
               :: "l"(map), "r"(smem_u32(src)), "r"(c0), "r"(c1) : "memory");
}

extern "C" __global__ void __launch_bounds__(256, 1)
transition_fwd_fp8(const __grid_constant__ CUtensorMap mx, const __grid_constant__ CUtensorMap mw1,
                   const __grid_constant__ CUtensorMap mw2, const __grid_constant__ CUtensorMap mout,
                   const float* __restrict__ gamma, const float* __restrict__ beta, const float* __restrict__ scales,
                   unsigned int* __restrict__ peak, int tiles, float eps) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  const int tid = threadIdx.x, wg = tid >> 7, wtid = tid & 127, warp = wtid >> 5, lane = tid & 31;
  const int cta = blockIdx.x;
  uint64_t* bars = reinterpret_cast<uint64_t*>(sm + F_BAR);
  uint64_t* w_full = bars; uint64_t* w_free = bars + NSLOT; uint64_t* x_full = bars + 2 * NSLOT; uint64_t* x_free = bars + 2 * NSLOT + 2;
  uint32_t* w_cnt = reinterpret_cast<uint32_t*>(sm + F_BAR + 512);
  const int n_local = (tiles > cta) ? (tiles - cta + NCTA - 1) / NCTA : 0;
  const uint32_t wmax = (uint32_t)n_local * NCH;
  const bool pp = tiles <= 2 * NCTA;
  const float sx = scales[0], ca = scales[1], cb = scales[2], co = scales[3];
  const float hca = 0.5f * ca;
  float pk = 0.f;

  auto issue_w = [&](uint32_t seq) {
    const int s = (int)(seq % NSLOT), j = (int)(seq % NCH); uint8_t* b = sm + s * F_SLOT;
    mbar_arrive_expect_tx(w_full + s, W_BYTES);
#if F8G1
    tma_load_2d(b, &mw1, w_full + s, 0, j * 128);                      // [Wa8_j; Wb8_j], 128 rows x 128 B
#else
    tma_load_2d(b, &mw1, w_full + s, 0, j * 128);                      // [Wa_j; Wb_j] bf16, columns 0..63, 128 rows
    tma_load_2d(b + 16384, &mw1, w_full + s, 64, j * 128);             // columns 64..127
#endif
#if F8G2
    tma_load_2d(b + F_W2, &mw2, w_full + s, j * HS, 0);               // Ws8[:, 64 j .. 64 j + 63], 128 rows x 64 B
#else
    tma_load_2d(b + F_W2, &mw2, w_full + s, 0, j * HS);               // Ws^T_j, [64 hs][128 d] as two 64-col boxes
    tma_load_2d(b + F_W2 + 8192, &mw2, w_full + s, 64, j * HS);
#endif
  };
  auto issue_x = [&](int i) {
    const int buf = i & 1, row = (cta + i * NCTA) * ROWS; uint8_t* b = sm + F_X + buf * F_XB;
    mbar_arrive_expect_tx(x_full + buf, 32768);
#pragma unroll
    for (int c = 0; c < 2; ++c)
#pragma unroll
      for (int h = 0; h < 2; ++h) tma_load_2d(b + c * 16384 + h * 8192, &mx, x_full + buf, c * 64, row + h * 64);
  };
  if (tid == 0) {
    tma_prefetch_desc(&mx); tma_prefetch_desc(&mw1); tma_prefetch_desc(&mw2); tma_prefetch_desc(&mout);
    for (int k = 0; k < NSLOT; ++k) { mbar_init(w_full + k, 1); mbar_init(w_free + k, 2); w_cnt[k] = 0; }
    mbar_init(x_full, 1); mbar_init(x_full + 1, 1); mbar_init(x_free, 2); mbar_init(x_free + 1, 2);
    fence_barrier_init();
  }
  __syncthreads();
  uint32_t wseq = 0;
  if (tid == 0) {
    if (n_local > 0) issue_x(0);
    for (uint32_t k = 0; k < (uint32_t)NSLOT; ++k) if (wmax > k && ABL != 3) issue_w(k);
    if (n_local > 1) issue_x(1);
  }

  for (int i = 0; i < n_local; ++i) {
    const int buf = i & 1, trow = (cta + i * NCTA) * ROWS;
    const uint32_t xu = su + F_X + buf * F_XB;
    mbar_wait(x_full + buf, (i >> 1) & 1);
    // ---------------------------------------------------------------- LayerNorm in the A-fragment layout (the 2.5.12 LNFRAG code), quantized
#if F8G1
    uint32_t af[4][4];                                   // e4m3 k32 fragments of xn sx, K in the logical (permuted) order
#else
    uint32_t af[8][4];                                   // bf16 k16 fragments of xn (the 2.5.12 RS operand)
#endif
    {
      uint32_t xr[8][4];
      const int mi = lane >> 3, mrow = wg * WGR + 16 * warp + 8 * (mi & 1) + (lane & 7);
#pragma unroll
      for (int ks = 0; ks < 8; ++ks) {
        const int col = 8 * (2 * ks + (mi >> 1));
        ldsm_x4(xr[ks], xu + (col >> 6) * 16384 + swz128((uint32_t)mrow, (uint32_t)((col & 63) * 2)));
      }
      const int q = lane & 3;
      float s2[2], mean2[2], v2[2], rs2[2];
#pragma unroll
      for (int R = 0; R < 2; ++R) {
        float s_ = 0.f;
#pragma unroll
        for (int ks = 0; ks < 8; ++ks) s_ += (bf16lo(xr[ks][R]) + bf16hi(xr[ks][R])) + (bf16lo(xr[ks][R + 2]) + bf16hi(xr[ks][R + 2]));
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
          d = bf16lo(xr[ks][R]) - mean2[R]; q_ += d * d;
          d = bf16hi(xr[ks][R]) - mean2[R]; q_ += d * d;
          d = bf16lo(xr[ks][R + 2]) - mean2[R]; q_ += d * d;
          d = bf16hi(xr[ks][R + 2]) - mean2[R]; q_ += d * d;
        }
        v2[R] = q_;
      }
#pragma unroll
      for (int o = 1; o <= 2; o <<= 1)
#pragma unroll
        for (int R = 0; R < 2; ++R) v2[R] += __shfl_xor_sync(0xffffffffu, v2[R], o);
#pragma unroll
      for (int R = 0; R < 2; ++R) rs2[R] = rsqrtf(v2[R] * (1.f / D_) + eps) * sx;
#pragma unroll
      for (int ks = 0; ks < 8; ++ks) {
        float gq[4], bq[4];
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const float2 gg = *reinterpret_cast<const float2*>(gamma + 16 * ks + 8 * h + 2 * q);
          const float2 bb = *reinterpret_cast<const float2*>(beta + 16 * ks + 8 * h + 2 * q);
          gq[2 * h] = gg.x; gq[2 * h + 1] = gg.y; bq[2 * h] = bb.x * sx; bq[2 * h + 1] = bb.y * sx;
        }
#pragma unroll
        for (int R = 0; R < 2; ++R) {
          const uint32_t u0 = xr[ks][R], u1 = xr[ks][R + 2];
#if F8G1
          af[ks >> 1][2 * (ks & 1) + R] = f8x4((bf16lo(u0) - mean2[R]) * rs2[R] * gq[0] + bq[0], (bf16hi(u0) - mean2[R]) * rs2[R] * gq[1] + bq[1],
                                               (bf16lo(u1) - mean2[R]) * rs2[R] * gq[2] + bq[2], (bf16hi(u1) - mean2[R]) * rs2[R] * gq[3] + bq[3]);
#else
          af[ks][R] = pack_bf16((bf16lo(u0) - mean2[R]) * rs2[R] * gq[0] + bq[0], (bf16hi(u0) - mean2[R]) * rs2[R] * gq[1] + bq[1]);
          af[ks][R + 2] = pack_bf16((bf16lo(u1) - mean2[R]) * rs2[R] * gq[2] + bq[2], (bf16hi(u1) - mean2[R]) * rs2[R] * gq[3] + bq[3]);
#endif
        }
      }
    }
    float acc[64];
#pragma unroll
    for (int e = 0; e < 64; ++e) acc[e] = 0.f;
#pragma unroll
    for (int j = 0; j < (ABL == 3 ? 0 : NCH); ++j, ++wseq) {
      const int s = (int)(wseq % NSLOT);
      const uint32_t slot = su + s * F_SLOT;
      mbar_wait(w_full + s, (uint32_t)((wseq / NSLOT) & 1));
      float AB[64];
#pragma unroll
      for (int e = 0; e < 64; ++e) AB[e] = 0.f;
#pragma unroll
      for (int m = 0; m < (F8G1 ? 4 : 8); ++m) fence_regs(af[m]);
      fence_regs(AB); fence_regs(acc);
      if (pp && j == 0 && wg == 1) asm volatile("bar.sync 5, 256;\n" ::: "memory");   // SKEW: warpgroup 1 starts after warpgroup 0's first chunk
      wgmma_fence();
#pragma unroll
#if F8G1
      for (int m = 0; m < 4; ++m) mma128_f8(AB, af[m], dsc_(slot + m * 32, 16, 1024, 1), m > 0);
#else
      for (int ks = 0; ks < 8; ++ks) mma128_bf_k(AB, af[ks], dsc_(slot + (ks >> 2) * 16384 + (ks & 3) * 32, 16, 1024, 1), ks > 0);
#endif
      wgmma_commit();
      if (pp && j == 0 && wg == 0) asm volatile("bar.arrive 5, 256;\n" ::: "memory");
      wgmma_wait<1>(); fence_regs(acc);                   // also retires the previous chunk's squeeze
      if (j > 0 || i > 0) {                              // the previous chunk's slot is free in this warpgroup
        const int ps = (int)((wseq - 1) % NSLOT);
        if (pp) {
          if (wtid == 0) {
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
      wgmma_wait<0>(); fence_regs(AB);
#if F8G2
      uint32_t fh[2][4];                                  // e4m3 k32 fragments of h sh: C groups 4m .. 4m + 3 -> step m
#pragma unroll
      for (int m = 0; m < 2; ++m)
#pragma unroll
        for (int R = 0; R < 2; ++R)
#pragma unroll
          for (int hf = 0; hf < 2; ++hf) {
            const int g0 = 4 * m + 2 * hf, g1 = g0 + 1;
            float v[4];
#pragma unroll
            for (int u = 0; u < 4; ++u) {
              const int e = 4 * (u < 2 ? g0 : g1) + 2 * R + (u & 1);
#if ABL == 4
              v[u] = AB[e] * cb + AB[e + 32] * cb;
#else
#if SIGT
              const float hu = AB[e] * hca;                 // a / 2;  silu(a) = a sigmoid(a) = (a / 2)(1 + tanh(a / 2))
              float th;
              asm("tanh.approx.f32 %0, %1;\n" : "=f"(th) : "f"(hu));
              v[u] = fmaf(hu, th, hu) * (AB[e + 32] * cb);
#else
              const float a = AB[e] * ca;
              v[u] = a * sigm(a) * (AB[e + 32] * cb);
#endif
#endif
              pk = fmaxf(pk, fabsf(v[u]));
            }
            fh[m][2 * hf + R] = f8x4(v[0], v[1], v[2], v[3]);
          }
      fence_regs(acc); wgmma_fence();
#pragma unroll
      for (int m = 0; m < 2; ++m) mma128_f8(acc, fh[m], dsc_(slot + F_W2 + m * 32, 16, 512, 2), 1);
#else
      uint32_t fh[4][4];
#pragma unroll
      for (int g = 0; g < 8; ++g) {
        const int e0 = 4 * g, ks = g >> 1, o = (g & 1) ? 2 : 0;
#pragma unroll
        for (int r = 0; r < 2; ++r) {
          const float a0 = AB[e0 + 2 * r] * ca, a1 = AB[e0 + 2 * r + 1] * ca;
          const float b0 = AB[e0 + 32 + 2 * r] * cb, b1 = AB[e0 + 32 + 2 * r + 1] * cb;
#if ABL == 4
          fh[ks][o + r] = pack_bf16(a0 + b0, a1 + b1);
#else
          fh[ks][o + r] = pack_bf16(a0 * sigm(a0) * b0, a1 * sigm(a1) * b1);
#endif
        }
      }
      fence_regs(acc); wgmma_fence();
#pragma unroll
      for (int ks = 0; ks < 4; ++ks) mma128_bf_rs(acc, fh[ks], dsc_(slot + F_W2 + ks * 2048, 8192, 1024, 1));
#endif
      wgmma_commit();
    }
    wgmma_wait<0>(); fence_regs(acc);
    // ---------------------------------------------------------------- out = bf16(x + acc co) over the x tile, one TMA store per half
    {
      const int mi = lane >> 3, mrow = wg * WGR + 16 * warp + 8 * (mi & 1) + (lane & 7);
#pragma unroll
      for (int gp = 0; gp < 8; ++gp) {
        const int col = 8 * (2 * gp + (mi >> 1));
        const uint32_t ad = xu + (col >> 6) * 16384 + swz128((uint32_t)mrow, (uint32_t)((col & 63) * 2));
        uint32_t xr[4];
        ldsm_x4(xr, ad);
        const int g0 = 2 * gp, g1 = 2 * gp + 1;
        stsm_x4(ad, pack_bf16(fmaf(acc[4 * g0 + 0], co, bf16lo(xr[0])), fmaf(acc[4 * g0 + 1], co, bf16hi(xr[0]))),
                    pack_bf16(fmaf(acc[4 * g0 + 2], co, bf16lo(xr[1])), fmaf(acc[4 * g0 + 3], co, bf16hi(xr[1]))),
                    pack_bf16(fmaf(acc[4 * g1 + 0], co, bf16lo(xr[2])), fmaf(acc[4 * g1 + 1], co, bf16hi(xr[2]))),
                    pack_bf16(fmaf(acc[4 * g1 + 2], co, bf16lo(xr[3])), fmaf(acc[4 * g1 + 3], co, bf16hi(xr[3]))));
      }
    }
    fence_proxy_async();
    named_bar_sync(1 + wg, 128);
    if (wtid == 0) {
#pragma unroll
      for (int c = 0; c < 2; ++c) tma_store_2d(&mout, sm + F_X + buf * F_XB + c * 16384 + wg * 8192, c * 64, trow + wg * WGR);
      tma_store_commit();
      tma_store_wait_all();
      mbar_arrive(x_free + buf);
    }
    if (tid == 0 && i + 2 < n_local) { mbar_wait(x_free + buf, (i >> 1) & 1); issue_x(i + 2); }
  }
  // the guard: max |h sh| of this CTA (before the saturating cast) -> peak (float bits, non-negative: integer max = float max)
  unsigned int pb = __reduce_max_sync(0xffffffffu, __float_as_uint(pk));
  if (lane == 0 && pb) atomicMax(peak, pb);
}

// ================================================================================================ host
namespace {
using EncodeTiled = CUresult (*)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*, const cuuint64_t*, const cuuint32_t*,
                                 const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
EncodeTiled encoder() {
  static EncodeTiled fn = [] {
    void* p = nullptr;
    cudaDriverEntryPointQueryResult q{};
    cudaError_t e = cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &p, cudaEnableDefault, &q);
    if (e != cudaSuccess || p == nullptr) throw std::runtime_error("cuTensorMapEncodeTiled unavailable");
    return reinterpret_cast<EncodeTiled>(p);
  }();
  return fn;
}
using Key = std::array<uint64_t, 7>;
// 2-D descriptor, dims innermost-first (elements), element size `esz` bytes, swizzle 64 / 128 B
const CUtensorMap& tile_map(const torch::Tensor& t, uint32_t d_inner, uint32_t d_outer, uint32_t box_inner, uint32_t box_outer, int esz, int swz) {
  static std::map<Key, CUtensorMap> cache;
  static std::mutex lock;
  const Key key{reinterpret_cast<uint64_t>(t.data_ptr()), d_inner, d_outer, box_inner, box_outer, (uint64_t)esz, (uint64_t)swz};
  std::lock_guard<std::mutex> guard(lock);
  auto it = cache.find(key);
  if (it != cache.end()) return it->second;
  CUtensorMap map{};
  const cuuint64_t dims[2] = {d_inner, d_outer};
  const cuuint64_t strides[1] = {static_cast<cuuint64_t>(d_inner) * esz};
  const cuuint32_t box[2] = {box_inner, box_outer};
  const cuuint32_t elem[2] = {1, 1};
  if (cudaSetDevice(t.get_device()) != cudaSuccess) throw std::runtime_error("cudaSetDevice failed");
  CUresult r = encoder()(&map, esz == 1 ? CU_TENSOR_MAP_DATA_TYPE_UINT8 : CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, t.data_ptr(), dims, strides, box, elem,
                         CU_TENSOR_MAP_INTERLEAVE_NONE, swz == 64 ? CU_TENSOR_MAP_SWIZZLE_64B : CU_TENSOR_MAP_SWIZZLE_128B,
                         CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (r != CUDA_SUCCESS) throw std::runtime_error("cuTensorMapEncodeTiled failed");
  return cache.emplace(key, map).first->second;
}
void expect(bool ok, const char* what) {
  if (!ok) throw std::invalid_argument(std::string("transition fp8 sm90a: ") + what);
}
}  // namespace

// x [M,128] bf16 (M a multiple of 128); w1 [NCH*128, 128] e4m3 (uint8 view; chunk j = [Wa_j; Wb_j], K permuted);
// w2: F8G2 Ws8 [128, H] e4m3 (K permuted) | else Ws^T [H, 128] bf16; scales float[8] = {sx, ca, cb, co}; peak uint32[1] (float bits, atomicMax).
torch::Tensor transition_fp8_fwd(torch::Tensor x, torch::Tensor gamma, torch::Tensor beta, torch::Tensor w1, torch::Tensor w2,
                                 torch::Tensor scales, torch::Tensor peak, double eps) {
  const int64_t M = x.size(0);
  expect(x.is_cuda() && x.is_contiguous() && x.scalar_type() == torch::kBFloat16 && x.size(1) == D_, "x must be contiguous cuda bf16 [M, 128]");
  expect(M % ROWS == 0, "row count must be a whole 128-row tile");
#if F8G1
  expect(w1.scalar_type() == torch::kUInt8 && w1.is_contiguous() && w1.size(0) == NCH * 128 && w1.size(1) == D_, "w1 must be uint8 [H/64*128, 128]");
#else
  expect(w1.scalar_type() == torch::kBFloat16 && w1.is_contiguous() && w1.size(0) == NCH * 128 && w1.size(1) == D_, "w1 must be bf16 [H/64*128, 128]");
#endif
#if F8G2
  expect(w2.scalar_type() == torch::kUInt8 && w2.is_contiguous() && w2.size(0) == D_ && w2.size(1) == H_, "w2 must be uint8 [128, H]");
#else
  expect(w2.scalar_type() == torch::kBFloat16 && w2.is_contiguous() && w2.size(0) == H_ && w2.size(1) == D_, "w2 must be bf16 [H, 128]");
#endif
  expect(gamma.scalar_type() == torch::kFloat32 && beta.scalar_type() == torch::kFloat32, "gamma/beta must be fp32");
  expect(scales.scalar_type() == torch::kFloat32 && scales.numel() >= 4, "scales float[>=4]");
  expect(peak.scalar_type() == torch::kFloat32 && peak.numel() >= 1, "peak float[1]");
  auto out = torch::empty_like(x);
  const int tiles = (int)(M / ROWS);
  static const bool ready = [] {
    cudaError_t e = cudaFuncSetAttribute(reinterpret_cast<const void*>(transition_fwd_fp8), cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES);
    if (e != cudaSuccess) throw std::runtime_error(std::string("transition_fwd_fp8 smem opt-in: ") + cudaGetErrorString(e));
    return true;
  }();
  (void)ready;
  if (tiles == 0) return out;
  const CUtensorMap& mx = tile_map(x, D_, (uint32_t)M, 64, 64, 2, 128);
#if F8G1
  const CUtensorMap& m1 = tile_map(w1, D_, NCH * 128, 128, 128, 1, 128);
#else
  const CUtensorMap& m1 = tile_map(w1, D_, NCH * 128, 64, 128, 2, 128);
#endif
#if F8G2
  const CUtensorMap& m2 = tile_map(w2, H_, D_, 64, 128, 1, 64);
#else
  const CUtensorMap& m2 = tile_map(w2, D_, H_, 64, 64, 2, 128);
#endif
  const CUtensorMap& mo = tile_map(out, D_, (uint32_t)M, 64, 64, 2, 128);
  transition_fwd_fp8<<<(tiles < NCTA ? tiles : NCTA), 256, SMEM_BYTES, at::cuda::getCurrentCUDAStream()>>>(
      mx, m1, m2, mo, gamma.data_ptr<float>(), beta.data_ptr<float>(), scales.data_ptr<float>(),
      reinterpret_cast<unsigned int*>(peak.data_ptr<float>()), tiles, (float)eps);
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("transition_fp8_fwd", &transition_fp8_fwd, "FP8 (e4m3) D128 Transition inference forward (opt-in)");
}
