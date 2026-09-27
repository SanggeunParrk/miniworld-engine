// transition_fwd_d256.cu -- the whole Transition forward (LayerNorm + SwiGLU expand + squeeze + residual) at D = 256,
// H = 1024, bf16, as ONE sm_90a kernel.  SPDX-License-Identifier: Apache-2.0
//
// The D = 128 kernel's row split, re-budgeted for D = 256:  a 128-row tile, two consumer warpgroups of 64 rows each, and the
// output accumulator of a warpgroup (64 x 256 fp32 = 128 registers) held across all 32 hidden chunks of 32 units; per chunk
//   [a|b] = xn [Wa_j; Wb_j]^T    m64n64 SS over K = 256 (16 k-steps)
//   h_j   = bf16(silu(a) b)      in the m64k32 A-fragment layout
//   acc  += h_j Ws^T_j           m64n256 RS over K = 32
// A producer warpgroup (setmaxnreg 40, consumers 232) feeds a THREE-slot weight ring (48 KB per chunk: one chunk of compute is
// shorter than one 48 KB L2 fetch, so two slots would expose it) and cycles ONE 64 KB tile buffer through
//   x(i) -> LayerNorm in place -> xn(i) (read by every G1) -> x(i) again (for the residual) -> out(i) staged -> x(i+1).
// Shared memory 208 KB.  sigmoid via tanh.approx (one MUFU); xn / rstd / c1 for the backward behind a runtime `save`.
#ifndef PP
#define PP 0
#endif
#include "tmn_kernels.cuh"
using namespace tmn; using namespace tmn::sm90;
#include "wgmma256rs.cuh"

#ifndef NCTA
#define NCTA 132
#endif
#ifndef NSLOT
#define NSLOT 3
#endif
#ifndef SIG_TANH
#define SIG_TANH 1
#endif
constexpr int D_ = 256, H_ = 1024, HS = 32, NCH = H_ / HS, ROWS = 128, WGR = 64;
constexpr int F_ABW = 32768, F_SLOT = 49152;                    // slot: [Wa_j; Wb_j] 4 quarters x [64 n][128 B], Ws^T_j at +32 KB
constexpr int F_XB = NSLOT * F_SLOT;                              // the tile buffer: 4 quarters x [128 rows][128 B]
constexpr int F_BAR = F_XB + 65536;
constexpr int SMEM_BYTES = F_BAR + 256;
static_assert(SMEM_BYTES <= 231424, "shared memory budget");

constexpr int PP_T0 = 3, PP_T1 = 4;                  // ping-pong turn barriers (1, 2 are the per-warpgroup syncs)
TMN_DEVI void named_bar_arrive_(int id, int n) { __syncwarp(); asm volatile("bar.arrive %0, %1;\n" :: "r"(id), "r"(n) : "memory"); }
TMN_DEVI float sigmoid_(float a) {
#if SIG_TANH
  float t; asm("tanh.approx.f32 %0, %1;" : "=f"(t) : "f"(0.5f * a)); return fmaf(0.5f, t, 0.5f);
#else
  return rcpf(__fadd_rn(1.f, ex2f(__fmul_rn(-1.4426950408889634f, a))));
#endif
}
TMN_DEVI uint64_t dsc_(uint32_t addr, uint32_t lbo, uint32_t sbo) { return smem_desc(addr, lbo, sbo, 1); }
TMN_DEVI void mma64(float (&d)[32], uint64_t a, uint64_t b, int accumulate) {
  asm volatile("{ .reg .pred p; setp.ne.b32 p, %34, 0; wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, %32, %33, p, 1, 1, 0, 0; }"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]) : "l"(a), "l"(b), "r"(accumulate));
}
TMN_DEVI void tma_store_2d(const CUtensorMap* map, const void* src, int c0, int c1) {
  asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%2, %3}], [%1];"
               :: "l"(map), "r"(smem_u32(src)), "r"(c0), "r"(c1) : "memory");
}
TMN_DEVI void stg128u(void* p, uint4 v) { asm volatile("st.global.v4.b32 [%0], {%1,%2,%3,%4};" :: "l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory"); }

extern "C" __global__ void __launch_bounds__(384, 1)
transition_fwd_fused(const __grid_constant__ CUtensorMap mx, const __grid_constant__ CUtensorMap mwa,
                     const __grid_constant__ CUtensorMap mwb, const __grid_constant__ CUtensorMap mwst,
                     const __grid_constant__ CUtensorMap mout, const float* __restrict__ gamma, const float* __restrict__ beta,
                     __nv_bfloat16* __restrict__ xn, __nv_bfloat16* __restrict__ out,
                     float* __restrict__ rstd, float* __restrict__ c1, int M, int tiles, float eps, int save) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  const int tid = threadIdx.x, wg = tid >> 7, wtid = tid & 127, warp = wtid >> 5, lane = tid & 31;
  const int cta = blockIdx.x;
  uint64_t* bars = reinterpret_cast<uint64_t*>(sm + F_BAR);
  uint64_t* w_full = bars; uint64_t* w_free = bars + NSLOT;
  uint64_t* x_load = bars + 2 * NSLOT;     // x(i) has landed (for the LayerNorm)
  uint64_t* xn_free = x_load + 1;          // both warpgroups' last G1 of the tile has retired: the buffer may take x(i) again
  uint64_t* x_res = x_load + 2;            // x(i) has landed again (for the residual)
  uint64_t* out_done = x_load + 3;         // both warpgroups' output stores have read the buffer
  const int n_local = (tiles > cta) ? (tiles - cta + NCTA - 1) / NCTA : 0;
  if (tid == 0) {
    for (int s = 0; s < NSLOT; ++s) { mbar_init(w_full + s, 1); mbar_init(w_free + s, 2); }
    mbar_init(x_load, 1); mbar_init(xn_free, 2); mbar_init(x_res, 1); mbar_init(out_done, 2);
    fence_barrier_init();
  }
  __syncthreads();
  if (wg == 0) {                                          // ------------------------------------------------ producer
    setmaxnreg_dec<40>();
    if (tid != 0) return;
    auto load_x = [&](int i, uint64_t* bar) {
      const int row = (cta + i * NCTA) * ROWS;
      mbar_arrive_expect_tx(bar, 65536);
      for (int q = 0; q < 4; ++q)
        for (int h = 0; h < 2; ++h) tma_load_2d(sm + F_XB + q * 16384 + h * 8192, &mx, bar, 64 * q, row + 64 * h);
    };
    auto load_w = [&](uint32_t seq) {                     // chunk seq % NCH into slot seq % NSLOT once its previous use retired
      const int s = (int)(seq % NSLOT), j = (int)(seq % NCH);
      if (seq >= NSLOT) mbar_wait(w_free + s, ((seq / NSLOT) - 1) & 1u);
      uint8_t* b = sm + s * F_SLOT;
      mbar_arrive_expect_tx(w_full + s, F_SLOT);
      for (int q = 0; q < 4; ++q) {
        tma_load_2d(b + q * 8192, &mwa, w_full + s, 64 * q, j * HS);            // rows 0..31 of the quarter: Wa_j
        tma_load_2d(b + q * 8192 + 4096, &mwb, w_full + s, 64 * q, j * HS);     // rows 32..63: Wb_j
        tma_load_2d(b + F_ABW + q * 4096, &mwst, w_full + s, 64 * q, j * HS);   // Ws^T_j, N-quarter q
      }
    };
    if (n_local == 0) return;
    load_x(0, x_load);
    for (int j = 0; j < NSLOT && j < NCH; ++j) load_w((uint32_t)j);
    for (int i = 0; i < n_local; ++i) {
      for (int j = NSLOT; j < NCH; ++j) load_w((uint32_t)(i * NCH + j));
      mbar_wait(xn_free, i & 1);                          // every G1 of tile i has retired
      load_x(i, x_res);                                   // x(i) again, for the residual
      if (i + 1 < n_local) for (int j = 0; j < NSLOT; ++j) load_w((uint32_t)((i + 1) * NCH + j));
      mbar_wait(out_done, i & 1);                         // the stores of out(i) have read the buffer
      if (i + 1 < n_local) load_x(i + 1, x_load);
    }
    return;
  }
  setmaxnreg_inc<232>();                                  // ------------------------------------------------ consumers
  const int cw = wg - 1;
  const uint32_t xb = su + F_XB;
  const int q_ = lane >> 3, gr_ = lane & 7;               // LayerNorm: lane owns 8 columns = granule gr_ of K-quarter q_
  uint32_t wseq = 0;
  for (int i = 0; i < n_local; ++i) {
    const int trow = (cta + i * NCTA) * ROWS;
    mbar_wait(x_load, i & 1);
    // ---------------------------------------------------------------- LayerNorm of this warpgroup's 64 rows, in place
    {
      float g8[8], b8[8];
      { const float4* gp = reinterpret_cast<const float4*>(gamma + 8 * lane); const float4* bp = reinterpret_cast<const float4*>(beta + 8 * lane);
        const float4 g0 = __ldg(gp), g1 = __ldg(gp + 1), b0 = __ldg(bp), b1 = __ldg(bp + 1);
        g8[0] = g0.x; g8[1] = g0.y; g8[2] = g0.z; g8[3] = g0.w; g8[4] = g1.x; g8[5] = g1.y; g8[6] = g1.z; g8[7] = g1.w;
        b8[0] = b0.x; b8[1] = b0.y; b8[2] = b0.z; b8[3] = b0.w; b8[4] = b1.x; b8[5] = b1.y; b8[6] = b1.z; b8[7] = b1.w; }
#pragma unroll 1
      for (int half = 0; half < 2; ++half) {
        uint4 v[8]; uint32_t ad[8]; float s8[8];
#pragma unroll
        for (int u = 0; u < 8; ++u) {
          const int r = WGR * cw + 16 * warp + 8 * half + u;
          ad[u] = xb + q_ * 16384 + swz128((uint32_t)r, (uint32_t)(gr_ * 16));
          v[u] = lds128(ad[u]);
        }
#pragma unroll
        for (int u = 0; u < 8; ++u)
          s8[u] = ((bf16lo(v[u].x) + bf16hi(v[u].x)) + (bf16lo(v[u].y) + bf16hi(v[u].y))) +
                  ((bf16lo(v[u].z) + bf16hi(v[u].z)) + (bf16lo(v[u].w) + bf16hi(v[u].w)));
#pragma unroll
        for (int k = 16; k; k >>= 1) {
#pragma unroll
          for (int u = 0; u < 8; ++u) s8[u] += __shfl_xor_sync(0xffffffffu, s8[u], k);
        }
        float mean8[8];
#pragma unroll
        for (int u = 0; u < 8; ++u) {
          mean8[u] = s8[u] * (1.f / D_);
          const uint32_t w4[4] = {v[u].x, v[u].y, v[u].z, v[u].w};
          float a = 0.f;
#pragma unroll
          for (int e = 0; e < 4; ++e) { float d = bf16lo(w4[e]) - mean8[u]; a += d * d; d = bf16hi(w4[e]) - mean8[u]; a += d * d; }
          s8[u] = a;
        }
#pragma unroll
        for (int k = 16; k; k >>= 1) {
#pragma unroll
          for (int u = 0; u < 8; ++u) s8[u] += __shfl_xor_sync(0xffffffffu, s8[u], k);
        }
#pragma unroll
        for (int u = 0; u < 8; ++u) {
          const int r = WGR * cw + 16 * warp + 8 * half + u;
          const float mean = mean8[u], rs = rsqrtf(s8[u] * (1.f / D_) + eps);
          const uint32_t w4[4] = {v[u].x, v[u].y, v[u].z, v[u].w};
          uint4 o;
          uint32_t* op = &o.x;
#pragma unroll
          for (int e = 0; e < 4; ++e)
            op[e] = pack_bf16((bf16lo(w4[e]) - mean) * rs * g8[2 * e] + b8[2 * e], (bf16hi(w4[e]) - mean) * rs * g8[2 * e + 1] + b8[2 * e + 1]);
          asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(ad[u]), "r"(o.x), "r"(o.y), "r"(o.z), "r"(o.w) : "memory");
          if (save) {
            stg128u(xn + (size_t)(trow + r) * D_ + 8 * lane, o);
            if (lane == 0) { rstd[trow + r] = rs; c1[trow + r] = mean * rs; }
          }
        }
      }
    }
    fence_proxy_async();                                  // xn (generic stores) -> the wgmma operand reads
    named_bar_sync(1 + cw, 128);
    // ---------------------------------------------------------------- the 32 hidden chunks
    float acc[128];
#pragma unroll
    for (int e = 0; e < 128; ++e) acc[e] = 0.f;
#if PP
    // Ping-pong: a warpgroup issues one batch [G2(j-1), G1(j)], hands the tensor pipe to the other warpgroup (named barrier), and
    // computes SwiGLU(j) while the other's batch runs.  WG0 skips its very first wait, WG1 its very last hand-over, so the
    // arrive / sync counts match.
    uint32_t fh[2][4];
    uint32_t pslot = 0;
    const bool last_tile = (i == n_local - 1);
#pragma unroll 1
    for (int j = 0; j <= NCH; ++j, ++wseq) {
      const int s = (int)(wseq % NSLOT);
      const uint32_t slot = su + s * F_SLOT;
      if (j < NCH) mbar_wait(w_full + s, (wseq / NSLOT) & 1u);
      float AB[32];
#pragma unroll
      for (int e = 0; e < 32; ++e) AB[e] = 0.f;
      if (!(cw == 0 && i == 0 && j == 0)) named_bar_sync(cw ? PP_T1 : PP_T0, 256);
      fence_regs(AB); fence_regs(acc); wgmma_fence();
      if (j > 0) {
        const uint64_t bd = dsc_(pslot + F_ABW, 4096, 1024);
        const uint32_t blo = (uint32_t)bd, bhi = (uint32_t)(bd >> 32);
        mma256_rs(acc, fh[0], blo, bhi, 0);
        mma256_rs(acc, fh[1], blo, bhi, 2048 >> 4);
      }
      if (j < NCH) {
#pragma unroll
        for (int ks = 0; ks < 16; ++ks)
          mma64(AB, dsc_(xb + (ks >> 2) * 16384 + cw * 8192 + (ks & 3) * 32, 16, 1024), dsc_(slot + (ks >> 2) * 8192 + (ks & 3) * 32, 16, 1024), ks > 0);
      }
      wgmma_commit();
      if (!(cw == 1 && last_tile && j == NCH)) named_bar_arrive_(cw ? PP_T0 : PP_T1, 256);
      wgmma_wait<0>(); fence_regs(acc); fence_regs(AB);
      if (j > 0 && wtid == 0) mbar_arrive(w_free + (int)((wseq - 1) % NSLOT));   // G2(j-1) has read chunk j-1's slot
      if (j == NCH) break;                                // (wseq not advanced past the tile's last chunk)
      if (j == NCH - 1 && wtid == 0) mbar_arrive(xn_free);
#pragma unroll
      for (int g = 0; g < 4; ++g) {
        const int e0 = 4 * g, ks = g >> 1, o = (g & 1) ? 2 : 0;
#pragma unroll
        for (int r = 0; r < 2; ++r) {
          const float a0 = AB[e0 + 2 * r], a1 = AB[e0 + 2 * r + 1];
          const float b0 = AB[e0 + 16 + 2 * r], b1 = AB[e0 + 16 + 2 * r + 1];
          fh[ks][o + r] = pack_bf16(a0 * sigmoid_(a0) * b0, a1 * sigmoid_(a1) * b1);
        }
      }
      pslot = slot;
    }
#else
#pragma unroll 1
    for (int j = 0; j < NCH; ++j, ++wseq) {
      const int s = (int)(wseq % NSLOT);
      const uint32_t slot = su + s * F_SLOT;
      mbar_wait(w_full + s, (wseq / NSLOT) & 1u);
      float AB[32];
#pragma unroll
      for (int e = 0; e < 32; ++e) AB[e] = 0.f;
      fence_regs(AB); fence_regs(acc); wgmma_fence();
#pragma unroll
      for (int ks = 0; ks < 16; ++ks)
        mma64(AB, dsc_(xb + (ks >> 2) * 16384 + cw * 8192 + (ks & 3) * 32, 16, 1024), dsc_(slot + (ks >> 2) * 8192 + (ks & 3) * 32, 16, 1024), ks > 0);
      wgmma_commit();
      wgmma_wait<1>(); fence_regs(acc);                   // retires the previous chunk's squeeze: its slot may be refilled
      if (j > 0 && wtid == 0) mbar_arrive(w_free + (int)((wseq - 1) % NSLOT));   // (the tile's last chunk is released after the loop)
      wgmma_wait<0>(); fence_regs(AB);
      if (j == NCH - 1 && wtid == 0) mbar_arrive(xn_free);  // this warpgroup's last read of xn(i) has retired
      uint32_t fh[2][4];                                  // h = bf16(silu(a) b): C groups 0..3 hold a, 4..7 hold b
#pragma unroll
      for (int g = 0; g < 4; ++g) {
        const int e0 = 4 * g, ks = g >> 1, o = (g & 1) ? 2 : 0;
#pragma unroll
        for (int r = 0; r < 2; ++r) {
          const float a0 = AB[e0 + 2 * r], a1 = AB[e0 + 2 * r + 1];
          const float b0 = AB[e0 + 16 + 2 * r], b1 = AB[e0 + 16 + 2 * r + 1];
          fh[ks][o + r] = pack_bf16(a0 * sigmoid_(a0) * b0, a1 * sigmoid_(a1) * b1);
        }
      }
      fence_regs(acc); wgmma_fence();
      {
        const uint64_t bd = dsc_(slot + F_ABW, 4096, 1024);          // Ws^T_j [K = 32 hs][N = 256 d]: N-atoms of 64 at +4 KB
        const uint32_t blo = (uint32_t)bd, bhi = (uint32_t)(bd >> 32);
        mma256_rs(acc, fh[0], blo, bhi, 0);
        mma256_rs(acc, fh[1], blo, bhi, 2048 >> 4);
      }
      wgmma_commit();
    }
    wgmma_wait<0>(); fence_regs(acc);
    if (wtid == 0) mbar_arrive(w_free + (int)((wseq - 1) % NSLOT));
#endif
    // ---------------------------------------------------------------- out = bf16(x + acc), in place over the reloaded x(i)
    mbar_wait(x_res, i & 1);
    {
      const int mi = lane >> 3, mrow = WGR * cw + 16 * warp + 8 * (mi & 1) + (lane & 7);
#pragma unroll
      for (int gp = 0; gp < 16; ++gp) {
        const int col = 8 * (2 * gp + (mi >> 1));
        const uint32_t ad = xb + (col >> 6) * 16384 + swz128((uint32_t)mrow, (uint32_t)((col & 63) * 2));
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
    named_bar_sync(1 + cw, 128);
    if (wtid == 0) {
      for (int q = 0; q < 4; ++q) tma_store_2d(&mout, sm + F_XB + q * 16384 + cw * 8192, 64 * q, trow + WGR * cw);
      tma_store_commit();
      tma_store_wait_read<0>();                            // the store has read the buffer: x(i+1) may land in it
      mbar_arrive(out_done);
    }
  }
  if (wtid == 0) tma_store_wait_all();
}
