// ln_swiglu_gemm_f8.cu -- OPT-IN FP8 (docs/fp8.md): ln_swiglu_gemm.cu writing h as e4m3 (h sh, saturating cast, max |h sh| to `peak`),
// the stored columns of each 16-wide block in the e4m3 k32 fragment order (stmatrix of e4m3 pairs; the squeeze B operand is packed in that
// order).  SWP build only; the expand GEMM stays bf16.  The original header follows.
// ln_swiglu_gemm.cu -- h = bf16( silu(LN(x) Wa^T) * LN(x) Wb^T ) with the LayerNorm INSIDE the GEMM (wide Transition widths).
// SPDX-License-Identifier: Apache-2.0
//
// A CTA owns a 128-row block: it TMA-loads the raw x rows, normalises them in place in shared memory (two-pass fp32 LN,
// each consumer warpgroup its own 64 rows), keeps that xn tile RESIDENT, and runs every N tile of the block against it,
// streaming only the weight slabs.  That removes the separate LN kernel, xn's HBM round trip (inference; with save = 1 it is
// written once for the backward) and every re-read of A from L2.  Tile 128 x 128 over W1p packed in 64-row blocks
// [Wa 64 | Wb 64], so one tile is 64 hidden units of a AND b; producer warpgroup (setmaxnreg 40 / 232); SwiGLU (tanh) from
// the fp32 accumulator; TMA-stored h.  KD = D at compile time (the resident tile is 128 x KD bf16).
#ifndef WAIT0
#define WAIT0 0
#endif
#include "tmn_kernels.cuh"
using namespace tmn; using namespace tmn::sm90;
#include "wgmma_n.cuh"
#include "wgmma8_n.cuh"
#ifndef F8G1
#define F8G1 0   // 1: the expand GEMM in e4m3 too (xn8 = e4m3(LN(x) sx) written in place over the x tile, W1p8 slabs [128 n][128 B], wgmma k32)
#endif

#ifndef KD
#define KD 512
#endif
constexpr int TBM = 128, TBN = 128, TBK = F8G1 ? 128 : 64, THB = 64, NQ = KD / 64, NK = KD / TBK;   // NK: K slabs of the expand
constexpr int F_A = 0, F_AB = TBM * KD * 2;                            // resident x / xn: NQ quarters x [128 rows][128 B]
constexpr int F_BS = 16384;                                            // B slab [128 n][64 k]
constexpr int F_STGW = 4096;                                           // per consumer warpgroup: h8 tile [64 rows][64 B], 64B swizzle
constexpr int NSTAGE = (231424 - 1024 - F_AB - 4 * F_STGW) / F_BS;
constexpr int F_B = F_AB, F_STG = F_B + NSTAGE * F_BS, F_BAR = F_STG + 4 * F_STGW;
constexpr int SMEM_BYTES = F_BAR + 256;
static_assert(NSTAGE >= 3 && SMEM_BYTES <= 231424, "shared memory budget");

TMN_DEVI float sigmoid_(float a) { float t; asm("tanh.approx.f32 %0, %1;" : "=f"(t) : "f"(0.5f * a)); return fmaf(0.5f, t, 0.5f); }
TMN_DEVI uint64_t dsc_(uint32_t addr, uint32_t lbo, uint32_t sbo) { return smem_desc(addr, lbo, sbo, 1); }
TMN_DEVI void tma_store_2d(const CUtensorMap* map, const void* src, int c0, int c1) {
  asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%2, %3}], [%1];"
               :: "l"(map), "r"(smem_u32(src)), "r"(c0), "r"(c1) : "memory");
}
TMN_DEVI void stg128u(void* p, uint4 v) { asm volatile("st.global.v4.b32 [%0], {%1,%2,%3,%4};" :: "l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory"); }


#ifndef SWP
#define SWP 0                                             // 1: software-pipelined epilogue -- tile n's SwiGLU epilogue runs, a quarter
#endif                                                    //    per slab, under tile n+1's MMAs (two 64-register accumulator sets)
struct LCtx { uint32_t su; uint64_t* full; uint64_t* empty; int cw, wtid, warp, lane; uint32_t it; float sh, pk, hca;
#if MW_SAVE_AB
  __nv_bfloat16* a; __nv_bfloat16* b; int h, row, n;
#endif
};
TMN_DEVI uint32_t f8x2_(float lo, float hi) { uint16_t r; asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;\n" : "=h"(r) : "f"(hi), "f"(lo)); return (uint32_t)r; }
template <int GP> TMN_DEVI void lepi_chunk(const float (&acc)[64], uint8_t* stg, LCtx& c) {
  const int mi = c.lane >> 3, mrow = 16 * c.warp + 8 * (mi & 1) + (c.lane & 7);
  uint32_t hq[2];
#pragma unroll
  for (int rb = 0; rb < 2; ++rb) {
    float v[4];
#pragma unroll
    for (int u = 0; u < 4; ++u) {
      const int ia = 4 * (2 * GP + (u >> 1)) + 2 * rb + (u & 1), ib = ia + 32;
      const float hu = c.hca * acc[ia];
      float th;
      asm("tanh.approx.f32 %0, %1;" : "=f"(th) : "f"(hu));
      v[u] = fmaf(hu, th, hu) * (acc[ib] * c.sh);          // silu(a) b sh, silu(a) = (a / 2)(1 + tanh(a / 2))
      c.pk = fmaxf(c.pk, fabsf(v[u]));
    }
    hq[rb] = f8x2_(v[0], v[1]) | (f8x2_(v[2], v[3]) << 16);
  }
  asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1,%2};\n" :: "r"(smem_u32(stg) + sw64((uint32_t)(mrow * 64 + 16 * GP))), "r"(hq[0]), "r"(hq[1]) : "memory");
}
TMN_DEVI void lepi_all(const float (&acc)[64], uint8_t* stg, LCtx& c) {
  lepi_chunk<0>(acc, stg, c); lepi_chunk<1>(acc, stg, c); lepi_chunk<2>(acc, stg, c); lepi_chunk<3>(acc, stg, c);
}
// all KB slabs of one N tile into A; when EPI, the previous tile's epilogue chunks run on P between the slabs
template <bool EPI> TMN_DEVI void lmain(float (&A)[64], const float (&P)[64], uint8_t* pstg, LCtx& c) {
#pragma unroll
  for (int e = 0; e < 64; ++e) A[e] = 0.f;
#pragma unroll
  for (int kb = 0; kb < NK; ++kb, ++c.it) {
    const int s = (int)(c.it % NSTAGE);
    mbar_wait(c.full + s, (c.it / NSTAGE) & 1u);
    const uint32_t a0 = c.su + F_A + kb * 16384 + c.cw * 8192, b0 = c.su + F_B + s * F_BS;
    fence_regs(A); wgmma_fence();
#pragma unroll
#if F8G1
    for (int ks = 0; ks < 4; ++ks) mma8_n128(A, smem_desc(a0 + ks * 32, 16, 1024, 1), smem_desc(b0 + ks * 32, 16, 1024, 1));
#else
    for (int ks = 0; ks < 4; ++ks) mma_n128(A, smem_desc(a0 + ks * 32, 16, 1024, 1), smem_desc(b0 + ks * 32, 16, 1024, 1));
#endif
    wgmma_commit();
    if constexpr (EPI) {
      if (kb == 0) lepi_chunk<0>(P, pstg, c);
      if (kb == NK / 4) lepi_chunk<1>(P, pstg, c);
      if (kb == NK / 2) lepi_chunk<2>(P, pstg, c);
      if (kb == 3 * NK / 4) lepi_chunk<3>(P, pstg, c);
    }
    wgmma_wait<0>(); fence_regs(A);
    if (c.wtid == 0) mbar_arrive(c.empty + s);
  }
}
TMN_DEVI void lepi_begin(const LCtx& c) {                 // the store that used this staging buffer two tiles ago has read it
  if (c.wtid == 0) tma_store_wait_read<1>();
  named_bar_sync(1 + c.cw, 128);
}
TMN_DEVI void lepi_end(const CUtensorMap* mH, uint8_t* stg, int col, int row, const LCtx& c) {
  fence_proxy_async();
  named_bar_sync(1 + c.cw, 128);
  if (c.wtid == 0) { tma_store_2d(mH, stg, col, row); tma_store_commit(); }
}

extern "C" __global__ void __launch_bounds__(384, 1)
ln_swiglu_gemm_f8(const __grid_constant__ CUtensorMap mX, const __grid_constant__ CUtensorMap mB, const __grid_constant__ CUtensorMap mH,
               const float* __restrict__ gamma, const float* __restrict__ beta, __nv_bfloat16* __restrict__ xn_out,
               float* __restrict__ rstd, float* __restrict__ c1, int M, int H, float eps, int save,
               const float* __restrict__ scales, unsigned int* __restrict__ peak
#if MW_SAVE_AB
               , __nv_bfloat16* saved_a, __nv_bfloat16* saved_b
#endif
               ) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  const int tid = threadIdx.x, wg = tid >> 7, wtid = tid & 127, warp = wtid >> 5, lane = tid & 31;
  uint64_t* full = reinterpret_cast<uint64_t*>(sm + F_BAR);
  uint64_t* empty = full + NSTAGE;
  uint64_t* a_full = empty + NSTAGE;                        // the block's raw x has landed
  uint64_t* a_free = a_full + 1;                            // both warpgroups' last MMA of the block has retired
  const int NT = H / THB, MB = M / TBM, KB = NK;
  if (tid == 0) {
    for (int s = 0; s < NSTAGE; ++s) { mbar_init(full + s, 1); mbar_init(empty + s, 2); }
    mbar_init(a_full, 1); mbar_init(a_free, 2);
    fence_barrier_init();
  }
  __syncthreads();
  if (wg == 0) {                                          // ------------------------------------------------ producer
    setmaxnreg_dec<40>();
    if (tid != 0) return;
    uint32_t it = 0, bl = 0;
    for (int mb = blockIdx.x; mb < MB; mb += gridDim.x, ++bl) {
      if (bl > 0) mbar_wait(a_free, (bl - 1) & 1u);
      mbar_arrive_expect_tx(a_full, F_AB);
      for (int q = 0; q < NQ; ++q)
        for (int h = 0; h < 2; ++h) tma_load_2d(sm + F_A + q * 16384 + h * 8192, &mX, a_full, 64 * q, mb * TBM + 64 * h);
      for (int n = 0; n < NT; ++n)
        for (int kb = 0; kb < KB; ++kb, ++it) {
          const int s = (int)(it % NSTAGE);
          if (it >= NSTAGE) mbar_wait(empty + s, ((it / NSTAGE) - 1) & 1u);
          mbar_arrive_expect_tx(full + s, F_BS);
          tma_load_2d(sm + F_B + s * F_BS, &mB, full + s, kb * TBK, n * TBN);
        }
    }
    return;
  }
  setmaxnreg_inc<232>();                                  // ------------------------------------------------ consumers
  const int cw = wg - 1;
  uint32_t it = 0, bl = 0, stc = 0;                       // stc: staging buffer toggle (two per warpgroup)
  float pk_all = 0.f;
  const float sxs = scales[0];   // xn scale (F8G1)
  (void)sxs;
  for (int mb = blockIdx.x; mb < MB; mb += gridDim.x, ++bl) {
    const int m0 = mb * TBM;
    mbar_wait(a_full, bl & 1u);
    // ---------------------------------------------------------------- LayerNorm of this warpgroup's 64 rows, in place
    // lane owns the 16-byte granules lane, lane + 32, ... of a row (KD / 8 granules); warp w rows 16 w .. 16 w + 15
    constexpr int G = KD / 8, GPL = (G + 31) / 32;
#pragma unroll 1
    for (int rr = 0; rr < 16; rr += 4) {
      uint4 v[4][GPL]; float s4[4];
#pragma unroll
      for (int u = 0; u < 4; ++u) {
        const int r = 64 * cw + 16 * warp + rr + u;
        float s = 0.f;
#pragma unroll
        for (int m = 0; m < GPL; ++m) {
          const int g = lane + 32 * m;
          if (g < G) {
            v[u][m] = lds128(su + F_A + (g >> 3) * 16384 + swz128((uint32_t)r, (uint32_t)((g & 7) * 16)));
            s += ((bf16lo(v[u][m].x) + bf16hi(v[u][m].x)) + (bf16lo(v[u][m].y) + bf16hi(v[u][m].y))) +
                 ((bf16lo(v[u][m].z) + bf16hi(v[u][m].z)) + (bf16lo(v[u][m].w) + bf16hi(v[u][m].w)));
          }
        }
        s4[u] = s;
      }
#pragma unroll
      for (int k = 16; k; k >>= 1) {
#pragma unroll
        for (int u = 0; u < 4; ++u) s4[u] += __shfl_xor_sync(0xffffffffu, s4[u], k);
      }
      float mean4[4];
#pragma unroll
      for (int u = 0; u < 4; ++u) {
        mean4[u] = s4[u] * (1.f / KD);
        float a = 0.f;
#pragma unroll
        for (int m = 0; m < GPL; ++m) {
          if (lane + 32 * m < G) {
            const uint32_t w4[4] = {v[u][m].x, v[u][m].y, v[u][m].z, v[u][m].w};
#pragma unroll
            for (int e = 0; e < 4; ++e) { float d = bf16lo(w4[e]) - mean4[u]; a += d * d; d = bf16hi(w4[e]) - mean4[u]; a += d * d; }
          }
        }
        s4[u] = a;
      }
#pragma unroll
      for (int k = 16; k; k >>= 1) {
#pragma unroll
        for (int u = 0; u < 4; ++u) s4[u] += __shfl_xor_sync(0xffffffffu, s4[u], k);
      }
#pragma unroll
      for (int u = 0; u < 4; ++u) {
        const int r = 64 * cw + 16 * warp + rr + u;
        const float mean = mean4[u], rs = rsqrtf(s4[u] * (1.f / KD) + eps);
#pragma unroll
        for (int m = 0; m < GPL; ++m) {
          const int g = lane + 32 * m;
          if (g < G) {
            const float4 ga = __ldg(reinterpret_cast<const float4*>(gamma + 8 * g)), gb = __ldg(reinterpret_cast<const float4*>(gamma + 8 * g + 4));
            const float4 ba = __ldg(reinterpret_cast<const float4*>(beta + 8 * g)), bb = __ldg(reinterpret_cast<const float4*>(beta + 8 * g + 4));
            const float gg[8] = {ga.x, ga.y, ga.z, ga.w, gb.x, gb.y, gb.z, gb.w}, bt[8] = {ba.x, ba.y, ba.z, ba.w, bb.x, bb.y, bb.z, bb.w};
            const uint32_t w4[4] = {v[u][m].x, v[u][m].y, v[u][m].z, v[u][m].w};
#if F8G1
            uint32_t o8[2];
#pragma unroll
            for (int e = 0; e < 2; ++e) {
              float q[4];
#pragma unroll
              for (int k = 0; k < 2; ++k) {
                const uint32_t w = w4[2 * e + k];
                const int c0 = 4 * e + 2 * k;
                q[2 * k] = ((bf16lo(w) - mean) * rs * gg[c0] + bt[c0]) * sxs;
                q[2 * k + 1] = ((bf16hi(w) - mean) * rs * gg[c0 + 1] + bt[c0 + 1]) * sxs;
              }
              o8[e] = f8x2_(q[0], q[1]) | (f8x2_(q[2], q[3]) << 16);
            }
            const uint32_t ad = su + F_A + (g >> 4) * 16384 + swz128((uint32_t)r, (uint32_t)((8 * g) & 127));   // in place: row r only
            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(ad), "r"(o8[0]), "r"(o8[1]) : "memory");
#else
            uint4 o; uint32_t* op = &o.x;
#pragma unroll
            for (int e = 0; e < 4; ++e)
              op[e] = pack_bf16((bf16lo(w4[e]) - mean) * rs * gg[2 * e] + bt[2 * e], (bf16hi(w4[e]) - mean) * rs * gg[2 * e + 1] + bt[2 * e + 1]);
            const uint32_t ad = su + F_A + (g >> 3) * 16384 + swz128((uint32_t)r, (uint32_t)((g & 7) * 16));
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(ad), "r"(o.x), "r"(o.y), "r"(o.z), "r"(o.w) : "memory");
            if (save) stg128u(xn_out + (size_t)(m0 + r) * KD + 8 * g, o);
#endif
          }
        }
        if (save && lane == 0) { rstd[m0 + r] = rs; c1[m0 + r] = mean * rs; }
      }
    }
    fence_proxy_async();
    named_bar_sync(1 + cw, 128);
    // ---------------------------------------------------------------- every N tile of the block against the resident xn
#if SWP
    {
      LCtx c{su, full, empty, cw, wtid, warp, lane, it, scales[2], 0.f, 0.5f * scales[1]};
#if MW_SAVE_AB
      c.a=save?saved_a:nullptr;c.b=save?saved_b:nullptr;c.h=H;c.row=m0+64*cw;c.n=0;
#endif
      float AX[64], AY[64];
      static_assert(NK >= 3, "the epilogue is spread over >= 3 slabs");
      lmain<false>(AX, AY, nullptr, c);                   // tile 0 -> X
      for (int n = 0; n < NT; n += 2) {                   // NT = H / 64 is even at every width built here
        uint8_t* sx = sm + F_STG + (cw * 2 + (int)(stc & 1)) * F_STGW;
        lepi_begin(c);
#if MW_SAVE_AB
        c.n=n;
#endif
        lmain<true>(AY, AX, sx, c);                       // tile n + 1 -> Y, epilogue of tile n (X) under it
        if (n + 1 == NT - 1 && wtid == 0) mbar_arrive(a_free);   // the last MMA of the block has read A
        lepi_end(&mH, sx, n * THB, m0 + 64 * cw, c); ++stc;
        uint8_t* sy = sm + F_STG + (cw * 2 + (int)(stc & 1)) * F_STGW;
        lepi_begin(c);
#if MW_SAVE_AB
        c.n=n+1;
#endif
        if (n + 2 < NT) lmain<true>(AX, AY, sy, c); else lepi_all(AY, sy, c);
        lepi_end(&mH, sy, (n + 1) * THB, m0 + 64 * cw, c); ++stc;
      }
      it = c.it;
      pk_all = fmaxf(pk_all, c.pk);
    }
#else
#error "ln_swiglu_gemm_f8: SWP build only"
    for (int n = 0; n < NT; ++n) {
      float acc[64];
#pragma unroll
      for (int e = 0; e < 64; ++e) acc[e] = 0.f;
      int prev = -1;
      for (int kb = 0; kb < KB; ++kb, ++it) {
        const int s = (int)(it % NSTAGE);
        mbar_wait(full + s, (it / NSTAGE) & 1u);
        const uint32_t a0 = su + F_A + kb * 16384 + cw * 8192, b0 = su + F_B + s * F_BS;
        fence_regs(acc); wgmma_fence();
#pragma unroll
        for (int ks = 0; ks < 4; ++ks) mma_n128(acc, dsc_(a0 + ks * 32, 16, 1024), dsc_(b0 + ks * 32, 16, 1024));
        wgmma_commit();
#if WAIT0                                                 // release this slab as soon as its MMAs retire
        wgmma_wait<0>(); fence_regs(acc); if (wtid == 0) mbar_arrive(empty + s);
#else
        if (prev >= 0) { wgmma_wait<1>(); fence_regs(acc); if (wtid == 0) mbar_arrive(empty + prev); }
        prev = s;
#endif
      }
      wgmma_wait<0>(); fence_regs(acc);
#if !WAIT0
      if (wtid == 0) mbar_arrive(empty + prev);
#endif
      if (n == NT - 1 && wtid == 0) mbar_arrive(a_free);   // the producer may bring the next block's x into A
      // ---------------------------------------------------------------- h = silu(a) b: a = columns 0..63, b = 64..127
      uint8_t* stg = sm + F_STG + (cw * 2 + (int)(stc & 1)) * F_STGW;
      if (wtid == 0) tma_store_wait_read<1>();             // the store that used this staging buffer two tiles ago has read it
      named_bar_sync(1 + cw, 128);
      {
        const int mi = lane >> 3, mrow = 16 * warp + 8 * (mi & 1) + (lane & 7);
#pragma unroll
        for (int gp = 0; gp < 4; ++gp) {
          uint32_t hp[4];
#pragma unroll
          for (int q = 0; q < 4; ++q) {
            const int g = 2 * gp + (q >> 1), rb = q & 1, ia = 4 * g + 2 * rb, ib = 4 * (g + 8) + 2 * rb;
            const float x0 = acc[ia], x1 = acc[ia + 1];
            hp[q] = pack_bf16(x0 * sigmoid_(x0) * acc[ib], x1 * sigmoid_(x1) * acc[ib + 1]);
          }
          const int col = 8 * (2 * gp + (mi >> 1));
          stsm_x4(smem_u32(stg) + swz128((uint32_t)mrow, (uint32_t)(col * 2)), hp[0], hp[1], hp[2], hp[3]);
        }
      }
      fence_proxy_async();
      named_bar_sync(1 + cw, 128);
      if (wtid == 0) { tma_store_2d(&mH, stg, n * THB, m0 + 64 * cw); tma_store_commit(); }
      ++stc;
    }
  #endif
}
  if (wtid == 0) tma_store_wait_all();
  const unsigned int pb = __reduce_max_sync(0xffffffffu, __float_as_uint(pk_all));
  if (lane == 0 && pb) atomicMax(peak, pb);
}
