// Bias-only token DiT, the HOISTED pair bias (integrations/bias_only_dit_hoist.py) in ONE pass over the pair each way, sm_90a, bf16 and fp32 (TF32).
//
//   hoist_fwd   bias^T [N, R] = W' LN0(pair)^T          LayerNorm rows (no affine) + the projection on the tensor cores; the normalised pair never
//                                                       exists in memory (it was written by a LayerNorm kernel and read back by cuBLAS)
//   hoist_bwd1  (N = 16: one block of 16 heads)         d LN0 = dbias^T W'  ->  d pair = rstd (d LN0 - mean(d LN0) - y mean(d LN0 y))
//                                                       dW' += dbias y   (bf16: y as the two-term split hi + lo; fp32: y in TF32)
//
// Both read the pair once, write the result once (forward: N values per row, backward: the pair gradient) and nothing else touches HBM: the
// bandwidth floor of the op. The structure that gets there:
//   * one WARP is a complete pipeline: it owns a ring of ST stages in shared memory (cp.async, 16-byte chunks, 16 pair rows = 4 KB / 8 KB per
//     stage), takes tiles of 16 consecutive rows (tile = warp + k * all_warps) and runs them with no block barrier: the warps of a CTA never
//     wait for each other, so every SM keeps ST x warps x 4-8 KB of loads in flight whatever the compute does;
//   * a tile is the M = 16 side of the mma: a row's 128 channels sit in 4 lanes (quad), so the LayerNorm statistics are in-register sums plus
//     two xor shuffles, the fragments of the pair tile come straight from ldmatrix, and the accumulator of d LN0 = dbias^T W' has the layout
//     of the tile it is applied to (no data movement between the matmul and the LayerNorm backward);
//   * dW' = dbias y needs y as the B operand [rows x channels]: bf16 gets it with movmatrix.trans of the register fragments (no shared
//     memory round trip), fp32 through a per-warp 8 KB tile.
// SYNC PROTOCOL. Inside a warp only (__syncwarp, cp.async groups): no cross-warp data flow until the last tile; then the warps' dW' partials
// (fp32, [16, CP]) are summed in a fixed order (warp 0 .. W-1) into one partial per CTA; the caller sums the CTAs' partials (a fixed order).
// No atomics. Every output element is written once; the result is a fixed function of the inputs and the (fixed) grid: bitwise repeatable.
//
// The pair width CP is a template parameter, 128 or 256. At CP = 256 the forward is the same code (twice the k-steps; the tile's row values
// fit in registers), with fewer pipeline stages where the 528 B rows would overflow shared memory. The backward cannot hold a 256-wide
// tile's d LN0 and y beside the dW' accumulators [16, 256] (128 registers per lane), so its CP = 256 form keeps only dW' resident and walks
// the tile one n-tile of 8 channels at a time, three times: the LayerNorm statistics, then d LN0 (one k16 mma per n-tile pair: cheap) for the
// two row sums, then d LN0 again for d pair (written in place over the tile) and the dW' products. The pair is still read once and d pair
// written once; the tile's values are re-read from shared memory.
//
// Numerics. bf16: LN0 with fp32 statistics, y rounded once to bf16 for the projection (the LayerNorm kernel's output, as before); d LN0 fp32;
// dW' against the split [bf16(y) | bf16(y - bf16(y))]. fp32: TF32 operands (round to nearest), fp32 accumulation, every tensor fp32.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include "bo_ptx.cuh"

namespace {
using bf = __nv_bfloat16;

using namespace bop;
__device__ __forceinline__ void mma16816(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void mma1688(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void ldsm4(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr) : "memory");
}
__device__ __forceinline__ void ldsm4t(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr) : "memory");
}
__device__ __forceinline__ uint32_t movtrans(uint32_t a) {
  uint32_t d;
  asm("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;" : "=r"(d) : "r"(a));
  return d;
}
__device__ __forceinline__ uint32_t tf32(float x) {
  uint32_t r;
  asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(x));
  return r;
}
__device__ __forceinline__ uint32_t pack2(float lo, float hi) {
  const __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<const uint32_t*>(&v);
}
__device__ __forceinline__ float bf_lo(uint32_t u) { return __uint_as_float(u << 16); }
__device__ __forceinline__ float bf_hi(uint32_t u) { return __uint_as_float(u & 0xffff0000u); }
__device__ __forceinline__ float rbf(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }

// LayerNorm statistics of the rows g and g + 8 of a tile, x[r][.] the lane's NV = CP / 4 channels of row g + 8 r (the quad t = 0..3 holds the row)
template <int NV>
__device__ __forceinline__ void row_stats(const float (&x)[2][NV], float eps, float (&mean)[2], float (&rstd)[2]) {
  constexpr float CP = 4.f * NV;
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    float s = 0.f;
#pragma unroll
    for (int i = 0; i < NV; ++i) s += x[r][i];
    s += __shfl_xor_sync(0xffffffffu, s, 1);
    s += __shfl_xor_sync(0xffffffffu, s, 2);
    mean[r] = s / CP;
    float q = 0.f;
#pragma unroll
    for (int i = 0; i < NV; ++i) { const float d = x[r][i] - mean[r]; q += d * d; }
    q += __shfl_xor_sync(0xffffffffu, q, 1);
    q += __shfl_xor_sync(0xffffffffu, q, 2);
    rstd[r] = rsqrtf(q / CP + eps);
  }
}
// the quad's sums of the two rows: in-register partials plus two xor shuffles
__device__ __forceinline__ void quad_sum2(float (&v)[2]) {
#pragma unroll
  for (int r = 0; r < 2; ++r) { v[r] += __shfl_xor_sync(0xffffffffu, v[r], 1); v[r] += __shfl_xor_sync(0xffffffffu, v[r], 2); }
}

// ============================================================================================================================================
// bf16
// ============================================================================================================================================
template <int CP> constexpr int XPB = CP + 8;                     // bf16 pair tile / weight pitch (elements): 272 / 528 B = an odd number of
                                                                   // 16 B units, ldmatrix and the quad stores are conflict free
constexpr int DPB = 24;                                            // dbias tile [16 heads][16 rows] pitch: 48 B
template <int CP> constexpr int XTILE_B = 16 * XPB<CP> * 2;      // 4352 / 8448 B
constexpr int DTILE_B = 16 * DPB * 2;                              // 768 B

template <int CP>
__device__ __forceinline__ void load_x_bf16(uint32_t xs, const bf* Z, long r0, int lane) {
#pragma unroll
  for (int q = 0; q < CP / 16; ++q) {
    const int i = lane + 32 * q, row = i / (CP / 8), c = i % (CP / 8);
    cp_async16(xs + (row * XPB<CP> + c * 8) * 2, Z + (r0 + row) * CP + c * 8);
  }
}

// the lane's pair-tile fragments -> xr[r][.] fp32 (row g + 8 r, channels 8 j + 2 t + e at xr[r][2 j + e], j = 0..CP / 8 - 1) and the raw words
template <int CP>
__device__ __forceinline__ void frag_x_bf16(uint32_t xs, int lane, uint32_t (&xa)[CP / 16][4], float (&xr)[2][CP / 4]) {
  const int row = (lane & 7) + ((lane >> 3) & 1) * 8, col = (lane >> 4) * 8;
#pragma unroll
  for (int ks = 0; ks < CP / 16; ++ks) {
    ldsm4(xs + (row * XPB<CP> + ks * 16 + col) * 2, xa[ks][0], xa[ks][1], xa[ks][2], xa[ks][3]);
    xr[0][4 * ks + 0] = bf_lo(xa[ks][0]); xr[0][4 * ks + 1] = bf_hi(xa[ks][0]);
    xr[1][4 * ks + 0] = bf_lo(xa[ks][1]); xr[1][4 * ks + 1] = bf_hi(xa[ks][1]);
    xr[0][4 * ks + 2] = bf_lo(xa[ks][2]); xr[0][4 * ks + 3] = bf_hi(xa[ks][2]);
    xr[1][4 * ks + 2] = bf_lo(xa[ks][3]); xr[1][4 * ks + 3] = bf_hi(xa[ks][3]);
  }
}

// ---------------------------------------------------------------- forward
template <int W, int ST, int CP>
__global__ void __launch_bounds__(W * 32, 1) fwd_bf16_k(const bf* __restrict__ Z, const bf* __restrict__ WF, bf* __restrict__ OUT, long R, int NB, float eps) {
  constexpr int KS = CP / 16, XP = XPB<CP>;
  extern __shared__ __align__(128) uint8_t sm[];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
  bf* wf = reinterpret_cast<bf*>(sm);                                   // [NB 16][XP]
  uint8_t* ring = sm + (size_t)NB * 16 * XP * 2 + (size_t)warp * ST * XTILE_B<CP>;
  for (int i = tid; i < NB * 16 * (CP / 8); i += W * 32) {
    const int h = i / (CP / 8), c = i % (CP / 8);
    *reinterpret_cast<uint4*>(wf + h * XP + c * 8) = *reinterpret_cast<const uint4*>(WF + (long)h * CP + c * 8);
  }
  __syncthreads();
  const long ntiles = R / 16, gw = (long)blockIdx.x * W + warp, step = (long)gridDim.x * W;
  const int nt = gw < ntiles ? (int)((ntiles - 1 - gw) / step) + 1 : 0;
#pragma unroll
  for (int s = 0; s < ST - 1; ++s) {
    if (s < nt) load_x_bf16<CP>(smem_u32(ring + s * XTILE_B<CP>), Z, (gw + (long)s * step) * 16, lane);
    cp_commit();
  }
  const uint32_t wfa = smem_u32(wf);
  for (int k = 0; k < nt; ++k) {
    cp_wait<ST - 2>();
    __syncwarp();
    if (k + ST - 1 < nt) load_x_bf16<CP>(smem_u32(ring + ((k + ST - 1) % ST) * XTILE_B<CP>), Z, (gw + (long)(k + ST - 1) * step) * 16, lane);
    cp_commit();
    const long r0 = (gw + (long)k * step) * 16;
    uint32_t xa[KS][4];
    float xr[2][CP / 4];
    frag_x_bf16<CP>(smem_u32(ring + (k % ST) * XTILE_B<CP>), lane, xa, xr);
    float mean[2], rstd[2];
    row_stats(xr, eps, mean, rstd);
    uint32_t ya[KS][4];                                                  // y rounded to bf16: the A operand
#pragma unroll
    for (int ks = 0; ks < KS; ++ks) {
      ya[ks][0] = pack2((xr[0][4 * ks + 0] - mean[0]) * rstd[0], (xr[0][4 * ks + 1] - mean[0]) * rstd[0]);
      ya[ks][1] = pack2((xr[1][4 * ks + 0] - mean[1]) * rstd[1], (xr[1][4 * ks + 1] - mean[1]) * rstd[1]);
      ya[ks][2] = pack2((xr[0][4 * ks + 2] - mean[0]) * rstd[0], (xr[0][4 * ks + 3] - mean[0]) * rstd[0]);
      ya[ks][3] = pack2((xr[1][4 * ks + 2] - mean[1]) * rstd[1], (xr[1][4 * ks + 3] - mean[1]) * rstd[1]);
    }
    for (int nb = 0; nb < NB; ++nb) {
      float acc[2][4] = {};
#pragma unroll
      for (int ks = 0; ks < KS; ++ks) {
        uint32_t b0, b1, b2, b3;
        ldsm4(wfa + ((nb * 16 + (lane & 7) + (lane >> 4) * 8) * XP + ks * 16 + ((lane >> 3) & 1) * 8) * 2, b0, b1, b2, b3);
        mma16816(acc[0], ya[ks][0], ya[ks][1], ya[ks][2], ya[ks][3], b0, b1);
        mma16816(acc[1], ya[ks][0], ya[ks][1], ya[ks][2], ya[ks][3], b2, b3);
      }
#pragma unroll
      for (int u = 0; u < 2; ++u) {
        bf* o = OUT + (long)(nb * 16 + u * 8 + 2 * t) * R + r0 + g;
        o[0] = __float2bfloat16_rn(acc[u][0]);
        o[R] = __float2bfloat16_rn(acc[u][1]);
        o[8] = __float2bfloat16_rn(acc[u][2]);
        o[R + 8] = __float2bfloat16_rn(acc[u][3]);
      }
    }
  }
}

// ---------------------------------------------------------------- backward (N = 16)
template <int CP> constexpr int BSTAGE_B = XTILE_B<CP> + DTILE_B;   // 5120 / 9216

template <int W, int ST, int CP>
__global__ void __launch_bounds__(W * 32, 1) bwd_bf16_k(const bf* __restrict__ Z, const bf* __restrict__ WF, const bf* __restrict__ DB, bf* __restrict__ DZ,
                                                        float* __restrict__ DWP, long R, float eps) {
  constexpr int NJ = CP / 8, XP = XPB<CP>;                            // n-tiles of 8 channels; tile pitch
  extern __shared__ __align__(128) uint8_t sm[];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
  bf* wf = reinterpret_cast<bf*>(sm);                                   // [16][XP]
  uint8_t* ring = sm + 16 * XP * 2 + (size_t)warp * ST * BSTAGE_B<CP>;
  for (int i = tid; i < 16 * (CP / 8); i += W * 32) {
    const int h = i / (CP / 8), c = i % (CP / 8);
    *reinterpret_cast<uint4*>(wf + h * XP + c * 8) = *reinterpret_cast<const uint4*>(WF + (long)h * CP + c * 8);
  }
  __syncthreads();
  const long ntiles = R / 16, gw = (long)blockIdx.x * W + warp, step = (long)gridDim.x * W;
  const int nt = gw < ntiles ? (int)((ntiles - 1 - gw) / step) + 1 : 0;
  auto load = [&](int s, long tile) {
    const uint32_t xs = smem_u32(ring + s * BSTAGE_B<CP>), ds = xs + XTILE_B<CP>;
    const long r0 = tile * 16;
    load_x_bf16<CP>(xs, Z, r0, lane);
    cp_async16(ds + ((lane >> 1) * DPB + (lane & 1) * 8) * 2, DB + (long)(lane >> 1) * R + r0 + (lane & 1) * 8);
  };
#pragma unroll
  for (int s = 0; s < ST - 1; ++s) {
    if (s < nt) load(s, gw + (long)s * step);
    cp_commit();
  }
  const uint32_t wfa = smem_u32(wf);
  float dw[NJ][4];
#pragma unroll
  for (int j = 0; j < NJ; ++j) dw[j][0] = dw[j][1] = dw[j][2] = dw[j][3] = 0.f;
  const int mi = lane >> 3;
  for (int k = 0; k < nt; ++k) {
    cp_wait<ST - 2>();
    __syncwarp();
    if (k + ST - 1 < nt) load((k + ST - 1) % ST, gw + (long)(k + ST - 1) * step);
    cp_commit();
    const long r0 = (gw + (long)k * step) * 16;
    uint8_t* stage = ring + (k % ST) * BSTAGE_B<CP>;
    const uint32_t xs = smem_u32(stage), ds = xs + XTILE_B<CP>;
    if constexpr (CP == 128) {
      uint32_t xa[8][4];
      float xr[2][32];
      frag_x_bf16<CP>(xs, lane, xa, xr);
      // d LN0 [16 rows, 128 ch] = dbias^T W'   (K = 16 heads)
      float dl[16][4];
#pragma unroll
      for (int j = 0; j < 16; ++j) dl[j][0] = dl[j][1] = dl[j][2] = dl[j][3] = 0.f;
      {
        uint32_t a0, a1, a2, a3;
        ldsm4t(ds + (((mi >> 1) * 8 + (lane & 7)) * DPB + (mi & 1) * 8) * 2, a0, a1, a2, a3);
#pragma unroll
        for (int np = 0; np < 8; ++np) {
          uint32_t b0, b1, b2, b3;
          ldsm4t(wfa + (((mi & 1) * 8 + (lane & 7)) * XP + np * 16 + (mi >> 1) * 8) * 2, b0, b1, b2, b3);
          mma16816(dl[2 * np], a0, a1, a2, a3, b0, b1);
          mma16816(dl[2 * np + 1], a0, a1, a2, a3, b2, b3);
        }
      }
      float mean[2], rstd[2];
      row_stats(xr, eps, mean, rstd);
      float y[2][32];
      float sd[2] = {0.f, 0.f}, sy[2] = {0.f, 0.f};
#pragma unroll
      for (int j = 0; j < 16; ++j)
#pragma unroll
        for (int r = 0; r < 2; ++r)
#pragma unroll
          for (int e = 0; e < 2; ++e) {
            const float yy = (xr[r][2 * j + e] - mean[r]) * rstd[r];
            y[r][2 * j + e] = yy;
            const float d = dl[j][2 * r + e];
            sd[r] += d;
            sy[r] = fmaf(d, yy, sy[r]);
          }
      float m1[2], m2[2];
#pragma unroll
      for (int r = 0; r < 2; ++r) {
        sd[r] += __shfl_xor_sync(0xffffffffu, sd[r], 1); sd[r] += __shfl_xor_sync(0xffffffffu, sd[r], 2);
        sy[r] += __shfl_xor_sync(0xffffffffu, sy[r], 1); sy[r] += __shfl_xor_sync(0xffffffffu, sy[r], 2);
        m1[r] = sd[r] / 128.f;
        m2[r] = sy[r] / 128.f;
      }
      // dW' += dbias y_hi + dbias y_lo (the next two lines of A operand and the B operands by movmatrix)
      uint32_t da0, da1, da2, da3;
      ldsm4(ds + (((lane & 7) + (mi & 1) * 8) * DPB + (mi >> 1) * 8) * 2, da0, da1, da2, da3);
#pragma unroll
      for (int j = 0; j < 16; ++j) {
        uint32_t hp[2], lp[2];
#pragma unroll
        for (int r = 0; r < 2; ++r) {
          const float h0 = rbf(y[r][2 * j]), h1 = rbf(y[r][2 * j + 1]);
          hp[r] = pack2(h0, h1);
          lp[r] = pack2(y[r][2 * j] - h0, y[r][2 * j + 1] - h1);
        }
        mma16816(dw[j], da0, da1, da2, da3, movtrans(hp[0]), movtrans(hp[1]));
        mma16816(dw[j], da0, da1, da2, da3, movtrans(lp[0]), movtrans(lp[1]));
      }
      // d pair, in place over the pair tile
      bf* xt = reinterpret_cast<bf*>(stage);
#pragma unroll
      for (int j = 0; j < 16; ++j)
#pragma unroll
        for (int r = 0; r < 2; ++r) {
          const float o0 = rstd[r] * (dl[j][2 * r] - m1[r] - y[r][2 * j] * m2[r]);
          const float o1 = rstd[r] * (dl[j][2 * r + 1] - m1[r] - y[r][2 * j + 1] * m2[r]);
          *reinterpret_cast<uint32_t*>(xt + (g + 8 * r) * XP + 8 * j + 2 * t) = pack2(o0, o1);
        }
    } else {
      // CP = 256: one n-tile pair (16 channels) at a time from the tile in shared memory (the lane's words of n-tile j: rows g, g + 8,
      // channels 8 j + 2 t, + 1 -- the d LN0 accumulator's positions); dW' stays in registers
      bf* xt = reinterpret_cast<bf*>(stage);
      auto xw = [&](int r, int j) { return *reinterpret_cast<const uint32_t*>(xt + (g + 8 * r) * XP + 8 * j + 2 * t); };
      float mean[2], rstd[2];
      {
        float s[2] = {0.f, 0.f}, q[2] = {0.f, 0.f};
#pragma unroll 8
        for (int j = 0; j < NJ; ++j)
#pragma unroll
          for (int r = 0; r < 2; ++r) { const uint32_t u = xw(r, j); s[r] += bf_lo(u); s[r] += bf_hi(u); }
        quad_sum2(s);
#pragma unroll
        for (int r = 0; r < 2; ++r) mean[r] = s[r] / (float)CP;
#pragma unroll 8
        for (int j = 0; j < NJ; ++j)
#pragma unroll
          for (int r = 0; r < 2; ++r) {
            const uint32_t u = xw(r, j);
            const float d0 = bf_lo(u) - mean[r], d1 = bf_hi(u) - mean[r];
            q[r] += d0 * d0; q[r] += d1 * d1;
          }
        quad_sum2(q);
#pragma unroll
        for (int r = 0; r < 2; ++r) rstd[r] = rsqrtf(q[r] / (float)CP + eps);
      }
      uint32_t a0, a1, a2, a3;                                          // A of d LN0: dbias^T
      ldsm4t(ds + (((mi >> 1) * 8 + (lane & 7)) * DPB + (mi & 1) * 8) * 2, a0, a1, a2, a3);
      auto dln2 = [&](int np, float (&d0)[4], float (&d1)[4]) {         // d LN0 of n-tiles 2 np, 2 np + 1 (K = 16 heads)
        uint32_t b0, b1, b2, b3;
        ldsm4t(wfa + (((mi & 1) * 8 + (lane & 7)) * XP + np * 16 + (mi >> 1) * 8) * 2, b0, b1, b2, b3);
        d0[0] = d0[1] = d0[2] = d0[3] = 0.f; d1[0] = d1[1] = d1[2] = d1[3] = 0.f;
        mma16816(d0, a0, a1, a2, a3, b0, b1);
        mma16816(d1, a0, a1, a2, a3, b2, b3);
      };
      float sd[2] = {0.f, 0.f}, sy[2] = {0.f, 0.f};
#pragma unroll 2
      for (int np = 0; np < NJ / 2; ++np) {
        float dl[2][4];
        dln2(np, dl[0], dl[1]);
#pragma unroll
        for (int jj = 0; jj < 2; ++jj)
#pragma unroll
          for (int r = 0; r < 2; ++r) {
            const uint32_t u = xw(r, 2 * np + jj);
            const float y0 = (bf_lo(u) - mean[r]) * rstd[r], y1 = (bf_hi(u) - mean[r]) * rstd[r];
            sd[r] += dl[jj][2 * r]; sd[r] += dl[jj][2 * r + 1];
            sy[r] = fmaf(dl[jj][2 * r], y0, sy[r]); sy[r] = fmaf(dl[jj][2 * r + 1], y1, sy[r]);
          }
      }
      quad_sum2(sd);
      quad_sum2(sy);
      float m1[2], m2[2];
#pragma unroll
      for (int r = 0; r < 2; ++r) { m1[r] = sd[r] / (float)CP; m2[r] = sy[r] / (float)CP; }
      uint32_t da0, da1, da2, da3;                                      // A of dW': dbias
      ldsm4(ds + (((lane & 7) + (mi & 1) * 8) * DPB + (mi >> 1) * 8) * 2, da0, da1, da2, da3);
#pragma unroll
      for (int np = 0; np < NJ / 2; ++np) {
        float dl[2][4];
        dln2(np, dl[0], dl[1]);
#pragma unroll
        for (int jj = 0; jj < 2; ++jj) {
          const int j = 2 * np + jj;
          uint32_t hp[2], lp[2];
#pragma unroll
          for (int r = 0; r < 2; ++r) {
            const uint32_t u = xw(r, j);
            const float y0 = (bf_lo(u) - mean[r]) * rstd[r], y1 = (bf_hi(u) - mean[r]) * rstd[r];
            const float o0 = rstd[r] * (dl[jj][2 * r] - m1[r] - y0 * m2[r]);
            const float o1 = rstd[r] * (dl[jj][2 * r + 1] - m1[r] - y1 * m2[r]);
            *reinterpret_cast<uint32_t*>(xt + (g + 8 * r) * XP + 8 * j + 2 * t) = pack2(o0, o1);   // d pair over the words just read
            const float h0 = rbf(y0), h1 = rbf(y1);
            hp[r] = pack2(h0, h1);
            lp[r] = pack2(y0 - h0, y1 - h1);
          }
          mma16816(dw[j], da0, da1, da2, da3, movtrans(hp[0]), movtrans(hp[1]));
          mma16816(dw[j], da0, da1, da2, da3, movtrans(lp[0]), movtrans(lp[1]));
        }
      }
    }
    __syncwarp();
    const bf* xt = reinterpret_cast<const bf*>(stage);
    bf* dz = DZ + r0 * CP;
#pragma unroll
    for (int q = 0; q < CP / 16; ++q) {
      const int i = lane + 32 * q, row = i / (CP / 8), c = i % (CP / 8);
      *reinterpret_cast<uint4*>(dz + (long)row * CP + c * 8) = *reinterpret_cast<const uint4*>(xt + row * XP + c * 8);
    }
  }
  // the warps' dW' partials, summed warp 0 .. W - 1
  cp_wait<0>();
  __syncthreads();
  float* red = reinterpret_cast<float*>(sm);                            // [W][16][CP]
#pragma unroll
  for (int j = 0; j < NJ; ++j) {
    *reinterpret_cast<float2*>(red + (warp * 16 + g) * CP + 8 * j + 2 * t) = make_float2(dw[j][0], dw[j][1]);
    *reinterpret_cast<float2*>(red + (warp * 16 + g + 8) * CP + 8 * j + 2 * t) = make_float2(dw[j][2], dw[j][3]);
  }
  __syncthreads();
  for (int e = tid; e < 16 * CP; e += W * 32) {
    float s = red[e];
#pragma unroll
    for (int w = 1; w < W; ++w) s += red[w * 16 * CP + e];
    DWP[(long)blockIdx.x * 16 * CP + e] = s;
  }
}

// ============================================================================================================================================
// fp32 (TF32 tensor cores)
// ============================================================================================================================================
template <int CP> constexpr int XPF = CP + 4;                     // fp32 pair tile pitch (floats): 528 / 1040 B = 4 CP + 16
template <int CP> constexpr int XTILE_F = 16 * XPF<CP> * 4;       // 8448 / 16640 B
template <int CP> constexpr int WPF_FWD = CP + 4;                 // weight pitch of the forward: bank (g * pitch + t) = 4 g + t
template <int CP> constexpr int WPF_BWD = CP + 8;                 // weight pitch of the backward: bank (t * pitch + col) = 8 t + col
constexpr int D1PF = 24;                                           // dbias tile for the scalar A operand of d LN0: bank 24 t + g covers all 32
constexpr int D2PF = 20;                                           // dbias tile for ldmatrix (A of dW'): 80 B rows
template <int CP> constexpr int YPF = CP == 128 ? 136 : 8;       // y tile for the B operand of dW': bank 8 t + g ([16][136]; CP = 256: one
                                                                   // n-tile at a time, [16][8])
template <int CP> constexpr int FSTAGE_B = XTILE_F<CP> + 16 * D1PF * 4 + 16 * D2PF * 4;   // 8448 + 1536 + 1280 = 11264 (19456)

template <int CP>
__device__ __forceinline__ void load_x_f32(uint32_t xs, const float* Z, long r0, int lane) {
#pragma unroll
  for (int q = 0; q < CP / 8; ++q) {
    const int i = lane + 32 * q, row = i / (CP / 4), c = i % (CP / 4);
    cp_async16(xs + (row * XPF<CP> + c * 4) * 4, Z + (r0 + row) * CP + c * 4);
  }
}
// fragments (TF32 A layout) of the pair tile: a0 = x[g][8 j + t], a1 = x[g + 8][8 j + t], a2 = x[g][8 j + t + 4], a3 = x[g + 8][8 j + t + 4];
// xr[r][2 j + e]: row g + 8 r, channel 8 j + t + 4 e
template <int CP>
__device__ __forceinline__ void frag_x_f32(uint32_t xs, int lane, float (&xr)[2][CP / 4]) {
  const int row = (lane & 7) + ((lane >> 3) & 1) * 8, col = (lane >> 4) * 4;
#pragma unroll
  for (int j = 0; j < CP / 8; ++j) {
    uint32_t a0, a1, a2, a3;
    ldsm4(xs + (row * XPF<CP> + 8 * j + col) * 4, a0, a1, a2, a3);
    xr[0][2 * j] = __uint_as_float(a0); xr[1][2 * j] = __uint_as_float(a1);
    xr[0][2 * j + 1] = __uint_as_float(a2); xr[1][2 * j + 1] = __uint_as_float(a3);
  }
}

template <int W, int ST, int CP>
__global__ void __launch_bounds__(W * 32, 1) fwd_f32_k(const float* __restrict__ Z, const float* __restrict__ WF, float* __restrict__ OUT, long R, int NB, float eps) {
  constexpr int WP = WPF_FWD<CP>;
  extern __shared__ __align__(128) uint8_t sm[];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
  uint32_t* wf = reinterpret_cast<uint32_t*>(sm);                       // [NB 16][WP], TF32 bits
  uint8_t* ring = sm + (size_t)NB * 16 * WP * 4 + (size_t)warp * ST * XTILE_F<CP>;
  for (int i = tid; i < NB * 16 * CP; i += W * 32) wf[(i / CP) * WP + (i % CP)] = tf32(WF[i]);
  __syncthreads();
  const long ntiles = R / 16, gw = (long)blockIdx.x * W + warp, step = (long)gridDim.x * W;
  const int nt = gw < ntiles ? (int)((ntiles - 1 - gw) / step) + 1 : 0;
#pragma unroll
  for (int s = 0; s < ST - 1; ++s) {
    if (s < nt) load_x_f32<CP>(smem_u32(ring + s * XTILE_F<CP>), Z, (gw + (long)s * step) * 16, lane);
    cp_commit();
  }
  for (int k = 0; k < nt; ++k) {
    cp_wait<ST - 2>();
    __syncwarp();
    if (k + ST - 1 < nt) load_x_f32<CP>(smem_u32(ring + ((k + ST - 1) % ST) * XTILE_F<CP>), Z, (gw + (long)(k + ST - 1) * step) * 16, lane);
    cp_commit();
    const long r0 = (gw + (long)k * step) * 16;
    float xr[2][CP / 4];
    frag_x_f32<CP>(smem_u32(ring + (k % ST) * XTILE_F<CP>), lane, xr);
    float mean[2], rstd[2];
    row_stats(xr, eps, mean, rstd);
    uint32_t ya[2][CP / 4];
#pragma unroll
    for (int r = 0; r < 2; ++r)
#pragma unroll
      for (int i = 0; i < CP / 4; ++i) ya[r][i] = tf32((xr[r][i] - mean[r]) * rstd[r]);
    for (int nb = 0; nb < NB; ++nb) {
      float acc[2][4] = {};
#pragma unroll
      for (int j = 0; j < CP / 8; ++j)
#pragma unroll
        for (int u = 0; u < 2; ++u) {
          const uint32_t* wr = wf + (nb * 16 + u * 8 + g) * WP + 8 * j + t;
          mma1688(acc[u], ya[0][2 * j], ya[1][2 * j], ya[0][2 * j + 1], ya[1][2 * j + 1], wr[0], wr[4]);
        }
#pragma unroll
      for (int u = 0; u < 2; ++u) {
        float* o = OUT + (long)(nb * 16 + u * 8 + 2 * t) * R + r0 + g;
        o[0] = acc[u][0];
        o[R] = acc[u][1];
        o[8] = acc[u][2];
        o[R + 8] = acc[u][3];
      }
    }
  }
}

template <int W, int ST, int CP>
__global__ void __launch_bounds__(W * 32, 1) bwd_f32_k(const float* __restrict__ Z, const float* __restrict__ WF, const float* __restrict__ DB, float* __restrict__ DZ,
                                                       float* __restrict__ DWP, long R, float eps) {
  constexpr int NJ = CP / 8, XP = XPF<CP>, WB = WPF_BWD<CP>, YP = YPF<CP>;
  extern __shared__ __align__(128) uint8_t sm[];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
  uint32_t* wf = reinterpret_cast<uint32_t*>(sm);                       // [16][WB], TF32 bits
  uint8_t* ring = sm + 16 * WB * 4 + (size_t)warp * (ST * FSTAGE_B<CP> + 16 * YP * 4);
  uint32_t* ytile = reinterpret_cast<uint32_t*>(ring + ST * FSTAGE_B<CP>);  // [16][YP] (n-column order)
  for (int i = tid; i < 16 * CP; i += W * 32) wf[(i / CP) * WB + (i % CP)] = tf32(WF[i]);
  __syncthreads();
  const long ntiles = R / 16, gw = (long)blockIdx.x * W + warp, step = (long)gridDim.x * W;
  const int nt = gw < ntiles ? (int)((ntiles - 1 - gw) / step) + 1 : 0;
  auto load = [&](int s, long tile) {
    const uint32_t xs = smem_u32(ring + s * FSTAGE_B<CP>), d1 = xs + XTILE_F<CP>, d2 = d1 + 16 * D1PF * 4;
    const long r0 = tile * 16;
    load_x_f32<CP>(xs, Z, r0, lane);
#pragma unroll
    for (int q = 0; q < 2; ++q) {
      const int i = lane + 32 * q, h = i >> 2, c = i & 3;
      const float* src = DB + (long)h * R + r0 + c * 4;
      cp_async16(d1 + (h * D1PF + c * 4) * 4, src);
      cp_async16(d2 + (h * D2PF + c * 4) * 4, src);
    }
  };
#pragma unroll
  for (int s = 0; s < ST - 1; ++s) {
    if (s < nt) load(s, gw + (long)s * step);
    cp_commit();
  }
  float dw[NJ][4];
#pragma unroll
  for (int j = 0; j < NJ; ++j) dw[j][0] = dw[j][1] = dw[j][2] = dw[j][3] = 0.f;
  const int mi = lane >> 3;
  for (int k = 0; k < nt; ++k) {
    cp_wait<ST - 2>();
    __syncwarp();
    if (k + ST - 1 < nt) load((k + ST - 1) % ST, gw + (long)(k + ST - 1) * step);
    cp_commit();
    const long r0 = (gw + (long)k * step) * 16;
    uint8_t* stage = ring + (k % ST) * FSTAGE_B<CP>;
    const uint32_t xs = smem_u32(stage), d2a = xs + XTILE_F<CP> + 16 * D1PF * 4;
    const float* d1 = reinterpret_cast<const float*>(stage + XTILE_F<CP>);
    float* xt = reinterpret_cast<float*>(stage);
    if constexpr (CP == 128) {
      float xr[2][32];
      frag_x_f32<CP>(xs, lane, xr);
      // d LN0 [16 rows, 128 ch] = dbias^T W'  (K = 16 heads = 2 k8 steps); tile j's column n holds channel 8 j + (n >> 1) + 4 (n & 1)
      float dl[16][4];
#pragma unroll
      for (int j = 0; j < 16; ++j) dl[j][0] = dl[j][1] = dl[j][2] = dl[j][3] = 0.f;
#pragma unroll
      for (int ks = 0; ks < 2; ++ks) {
        const uint32_t a0 = tf32(d1[(8 * ks + t) * D1PF + g]), a1 = tf32(d1[(8 * ks + t) * D1PF + g + 8]);
        const uint32_t a2 = tf32(d1[(8 * ks + t + 4) * D1PF + g]), a3 = tf32(d1[(8 * ks + t + 4) * D1PF + g + 8]);
        const int cg = (g >> 1) + 4 * (g & 1);
#pragma unroll
        for (int j = 0; j < 16; ++j)
          mma1688(dl[j], a0, a1, a2, a3, wf[(8 * ks + t) * WB + 8 * j + cg], wf[(8 * ks + t + 4) * WB + 8 * j + cg]);
      }
      float mean[2], rstd[2];
      row_stats(xr, eps, mean, rstd);
      float sd[2] = {0.f, 0.f}, sy[2] = {0.f, 0.f};
      float y[2][32];
#pragma unroll
      for (int j = 0; j < 16; ++j)
#pragma unroll
        for (int r = 0; r < 2; ++r)
#pragma unroll
          for (int e = 0; e < 2; ++e) {
            const float yy = (xr[r][2 * j + e] - mean[r]) * rstd[r];
            y[r][2 * j + e] = yy;
            const float d = dl[j][2 * r + e];
            sd[r] += d;
            sy[r] = fmaf(d, yy, sy[r]);
          }
      float m1[2], m2[2];
#pragma unroll
      for (int r = 0; r < 2; ++r) {
        sd[r] += __shfl_xor_sync(0xffffffffu, sd[r], 1); sd[r] += __shfl_xor_sync(0xffffffffu, sd[r], 2);
        sy[r] += __shfl_xor_sync(0xffffffffu, sy[r], 1); sy[r] += __shfl_xor_sync(0xffffffffu, sy[r], 2);
        m1[r] = sd[r] / 128.f;
        m2[r] = sy[r] / 128.f;
      }
      // y (TF32) to the tile in n-column order: row g + 8 r, column 8 j + 2 t + e
#pragma unroll
      for (int j = 0; j < 16; ++j)
#pragma unroll
        for (int r = 0; r < 2; ++r)
          *reinterpret_cast<uint2*>(ytile + (g + 8 * r) * YP + 8 * j + 2 * t) = make_uint2(tf32(y[r][2 * j]), tf32(y[r][2 * j + 1]));
      // d pair in place over the pair tile (the channel order of the fragments)
#pragma unroll
      for (int j = 0; j < 16; ++j)
#pragma unroll
        for (int r = 0; r < 2; ++r)
#pragma unroll
          for (int e = 0; e < 2; ++e)
            xt[(g + 8 * r) * XP + 8 * j + t + 4 * e] = rstd[r] * (dl[j][2 * r + e] - m1[r] - y[r][2 * j + e] * m2[r]);
      __syncwarp();
      // dW' += dbias y  (M = 16 heads, N = 128 columns, K = 16 rows = 2 k8 steps)
#pragma unroll
      for (int kk = 0; kk < 2; ++kk) {
        uint32_t a0, a1, a2, a3;
        ldsm4(d2a + (((lane & 7) + (mi & 1) * 8) * D2PF + kk * 8 + (mi >> 1) * 4) * 4, a0, a1, a2, a3);
        a0 = tf32(__uint_as_float(a0)); a1 = tf32(__uint_as_float(a1)); a2 = tf32(__uint_as_float(a2)); a3 = tf32(__uint_as_float(a3));
#pragma unroll
        for (int j = 0; j < 16; ++j)
          mma1688(dw[j], a0, a1, a2, a3, ytile[(8 * kk + t) * YP + 8 * j + g], ytile[(8 * kk + t + 4) * YP + 8 * j + g]);
      }
    } else {
      // CP = 256: one n-tile (8 channels) at a time from the tile in shared memory (the lane's values of n-tile j: rows g, g + 8, channels
      // 8 j + t + 4 e -- the d LN0 accumulator's positions); dW' stays in registers, y passes through a [16][8] tile per n-tile
      auto xv = [&](int r, int j, int e) { return xt[(g + 8 * r) * XP + 8 * j + t + 4 * e]; };
      float mean[2], rstd[2];
      {
        float s[2] = {0.f, 0.f}, q[2] = {0.f, 0.f};
#pragma unroll 8
        for (int j = 0; j < NJ; ++j)
#pragma unroll
          for (int r = 0; r < 2; ++r) { s[r] += xv(r, j, 0); s[r] += xv(r, j, 1); }
        quad_sum2(s);
#pragma unroll
        for (int r = 0; r < 2; ++r) mean[r] = s[r] / (float)CP;
#pragma unroll 8
        for (int j = 0; j < NJ; ++j)
#pragma unroll
          for (int r = 0; r < 2; ++r) {
            const float d0 = xv(r, j, 0) - mean[r], d1v = xv(r, j, 1) - mean[r];
            q[r] += d0 * d0; q[r] += d1v * d1v;
          }
        quad_sum2(q);
#pragma unroll
        for (int r = 0; r < 2; ++r) rstd[r] = rsqrtf(q[r] / (float)CP + eps);
      }
      uint32_t a[2][4];                                                 // A of d LN0: dbias^T, 2 k8 steps
#pragma unroll
      for (int ks = 0; ks < 2; ++ks) {
        a[ks][0] = tf32(d1[(8 * ks + t) * D1PF + g]); a[ks][1] = tf32(d1[(8 * ks + t) * D1PF + g + 8]);
        a[ks][2] = tf32(d1[(8 * ks + t + 4) * D1PF + g]); a[ks][3] = tf32(d1[(8 * ks + t + 4) * D1PF + g + 8]);
      }
      const int cg = (g >> 1) + 4 * (g & 1);
      auto dln = [&](int j, float (&d)[4]) {                            // d LN0 of n-tile j (K = 16 heads)
        d[0] = d[1] = d[2] = d[3] = 0.f;
#pragma unroll
        for (int ks = 0; ks < 2; ++ks)
          mma1688(d, a[ks][0], a[ks][1], a[ks][2], a[ks][3], wf[(8 * ks + t) * WB + 8 * j + cg], wf[(8 * ks + t + 4) * WB + 8 * j + cg]);
      };
      float sd[2] = {0.f, 0.f}, sy[2] = {0.f, 0.f};
#pragma unroll 4
      for (int j = 0; j < NJ; ++j) {
        float dl[4];
        dln(j, dl);
#pragma unroll
        for (int r = 0; r < 2; ++r)
#pragma unroll
          for (int e = 0; e < 2; ++e) {
            const float yy = (xv(r, j, e) - mean[r]) * rstd[r];
            sd[r] += dl[2 * r + e];
            sy[r] = fmaf(dl[2 * r + e], yy, sy[r]);
          }
      }
      quad_sum2(sd);
      quad_sum2(sy);
      float m1[2], m2[2];
#pragma unroll
      for (int r = 0; r < 2; ++r) { m1[r] = sd[r] / (float)CP; m2[r] = sy[r] / (float)CP; }
      uint32_t da[2][4];                                                // A of dW': dbias, 2 k8 steps
#pragma unroll
      for (int kk = 0; kk < 2; ++kk) {
        ldsm4(d2a + (((lane & 7) + (mi & 1) * 8) * D2PF + kk * 8 + (mi >> 1) * 4) * 4, da[kk][0], da[kk][1], da[kk][2], da[kk][3]);
#pragma unroll
        for (int i = 0; i < 4; ++i) da[kk][i] = tf32(__uint_as_float(da[kk][i]));
      }
#pragma unroll
      for (int j = 0; j < NJ; ++j) {
        float dl[4];
        dln(j, dl);
#pragma unroll
        for (int r = 0; r < 2; ++r) {
          float yv[2];
#pragma unroll
          for (int e = 0; e < 2; ++e) {
            const int at = (g + 8 * r) * XP + 8 * j + t + 4 * e;
            yv[e] = (xt[at] - mean[r]) * rstd[r];
            xt[at] = rstd[r] * (dl[2 * r + e] - m1[r] - yv[e] * m2[r]);  // d pair over the value just read
          }
          *reinterpret_cast<uint2*>(ytile + (g + 8 * r) * YP + 2 * t) = make_uint2(tf32(yv[0]), tf32(yv[1]));
        }
        __syncwarp();
#pragma unroll
        for (int kk = 0; kk < 2; ++kk)
          mma1688(dw[j], da[kk][0], da[kk][1], da[kk][2], da[kk][3], ytile[(8 * kk + t) * YP + g], ytile[(8 * kk + t + 4) * YP + g]);
        __syncwarp();                                                   // the y tile is rewritten for the next n-tile
      }
    }
    float* dz = DZ + r0 * CP;
#pragma unroll
    for (int q = 0; q < CP / 8; ++q) {
      const int i = lane + 32 * q, row = i / (CP / 4), c = i % (CP / 4);
      *reinterpret_cast<float4*>(dz + (long)row * CP + c * 4) = *reinterpret_cast<const float4*>(xt + row * XP + c * 4);
    }
    __syncwarp();                                                       // the y tile is rewritten next tile
  }
  cp_wait<0>();
  __syncthreads();
  float* red = reinterpret_cast<float*>(sm);                            // [W][16][CP], channel order
#pragma unroll
  for (int j = 0; j < NJ; ++j)
#pragma unroll
    for (int e = 0; e < 2; ++e) {
      const int ch = 8 * j + t + 4 * e;                                 // = 8 j + (n >> 1) + 4 (n & 1) at n = 2 t + e
      red[(warp * 16 + g) * CP + ch] = dw[j][e];
      red[(warp * 16 + g + 8) * CP + ch] = dw[j][2 + e];
    }
  __syncthreads();
  for (int e = tid; e < 16 * CP; e += W * 32) {
    float s = red[e];
#pragma unroll
    for (int w = 1; w < W; ++w) s += red[w * 16 * CP + e];
    DWP[(long)blockIdx.x * 16 * CP + e] = s;
  }
}

// ============================================================================================================================================
// launch
// ============================================================================================================================================
template <typename K> void set_smem(K k, size_t smem) {
  TORCH_CHECK(smem <= 232448, "hoist pass: shared memory ", smem);
  TORCH_CHECK(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem) == cudaSuccess, "hoist pass: smem attribute");
}
int num_sms() { return at::cuda::getCurrentDeviceProperties()->multiProcessorCount; }

void check_rows(const at::Tensor& z, const at::Tensor& w, int64_t& R, int64_t& N, int64_t& C) {
  TORCH_CHECK(z.is_cuda() && z.dim() == 2 && (z.size(1) == 128 || z.size(1) == 256) && z.is_contiguous(), "hoist pass: pair rows [R, 128 or 256]");
  TORCH_CHECK(z.scalar_type() == at::kBFloat16 || z.scalar_type() == at::kFloat, "hoist pass: bf16 or fp32");
  C = z.size(1);
  TORCH_CHECK(w.scalar_type() == z.scalar_type() && w.is_contiguous() && w.dim() == 2 && w.size(1) == C, "hoist pass: weights like the pair [N, C]");
  R = z.size(0); N = w.size(0);
  TORCH_CHECK(R % 16 == 0 && N % 16 == 0 && N >= 16, "hoist pass: R and N multiples of 16");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(z.data_ptr()) % 16 == 0, "hoist pass: 16-byte aligned pair");
}

#ifndef HP_FWD_W
#define HP_FWD_W 8
#endif
#ifndef HP_FWD_ST
#define HP_FWD_ST 3
#endif
#ifndef HP_FWD_ST32
#define HP_FWD_ST32 2
#endif
#ifndef HP_BWD_W
#define HP_BWD_W 8
#endif
#ifndef HP_BWD_ST
#define HP_BWD_ST 4
#endif
#ifndef HP_BWD_W32
#define HP_BWD_W32 4
#endif
#ifndef HP_BWD_ST32
#define HP_BWD_ST32 3
#endif
// warps per CTA and pipeline stages per warp (swept on an H100 at R = 4 L^2, L = 256 / 512: the forward is flat over 3-5 stages (bf16) / 2-3 (fp32), the backward over 3-5; the build flags -DHP_* override them)
constexpr int FWD_W = HP_FWD_W, FWD_ST = HP_FWD_ST, FWD_ST32 = HP_FWD_ST32, BWD_W = HP_BWD_W, BWD_ST = HP_BWD_ST, BWD_W32 = HP_BWD_W32, BWD_ST32 = HP_BWD_ST32;
// the 256-wide pair: two stages per warp (the 8.25 / 16.25 KB tiles; three overflow shared memory at 8 warps), 4 warps for the fp32 backward
constexpr int FWD_ST_256 = 2, BWD_ST_256 = 2, BWD_W32_256 = 4, BWD_ST32_256 = 2;

template <int CP>
void fwd_launch(const at::Tensor& z, const at::Tensor& wf, at::Tensor& out, int64_t R, int nb, double eps) {
  const int grid = num_sms();
  auto st = at::cuda::getCurrentCUDAStream();
  if (z.scalar_type() == at::kBFloat16) {
    constexpr int S = CP == 128 ? FWD_ST : FWD_ST_256;
    const size_t smem = (size_t)nb * 16 * XPB<CP> * 2 + (size_t)FWD_W * S * XTILE_B<CP>;
    auto k = fwd_bf16_k<FWD_W, S, CP>;
    set_smem(k, smem);
    k<<<grid, FWD_W * 32, smem, st>>>(reinterpret_cast<const bf*>(z.data_ptr()), reinterpret_cast<const bf*>(wf.data_ptr()), reinterpret_cast<bf*>(out.data_ptr()), R, nb, (float)eps);
  } else {
    const size_t wbytes = (size_t)nb * 16 * WPF_FWD<CP> * 4;
    auto run = [&](auto k, int w) {
      const size_t smem = wbytes + (size_t)w * FWD_ST32 * XTILE_F<CP>;
      set_smem(k, smem);
      k<<<grid, w * 32, smem, st>>>(reinterpret_cast<const float*>(z.data_ptr()), reinterpret_cast<const float*>(wf.data_ptr()), reinterpret_cast<float*>(out.data_ptr()), R, nb, (float)eps);
    };
    if (wbytes + (size_t)FWD_W * FWD_ST32 * XTILE_F<CP> <= 232448) run(fwd_f32_k<FWD_W, FWD_ST32, CP>, FWD_W);
    else run(fwd_f32_k<4, FWD_ST32, CP>, 4);                            // many weight rows (or the 256-wide pair): fewer warps per CTA
  }
}

template <int CP>
void bwd_launch(const at::Tensor& z, const at::Tensor& wf, const at::Tensor& db, at::Tensor& dz, at::Tensor& dwp, int64_t R, double eps) {
  const int grid = num_sms();
  auto st = at::cuda::getCurrentCUDAStream();
  if (z.scalar_type() == at::kBFloat16) {
    constexpr int S = CP == 128 ? BWD_ST : BWD_ST_256;
    const size_t smem = std::max((size_t)16 * XPB<CP> * 2 + (size_t)BWD_W * S * BSTAGE_B<CP>, (size_t)BWD_W * 16 * CP * 4);
    auto k = bwd_bf16_k<BWD_W, S, CP>;
    set_smem(k, smem);
    k<<<grid, BWD_W * 32, smem, st>>>(reinterpret_cast<const bf*>(z.data_ptr()), reinterpret_cast<const bf*>(wf.data_ptr()), reinterpret_cast<const bf*>(db.data_ptr()),
                                      reinterpret_cast<bf*>(dz.data_ptr()), reinterpret_cast<float*>(dwp.data_ptr()), R, (float)eps);
  } else {
    constexpr int W = CP == 128 ? BWD_W32 : BWD_W32_256, S = CP == 128 ? BWD_ST32 : BWD_ST32_256;
    const size_t smem = std::max((size_t)16 * WPF_BWD<CP> * 4 + (size_t)W * (S * FSTAGE_B<CP> + 16 * YPF<CP> * 4), (size_t)W * 16 * CP * 4);
    auto k = bwd_f32_k<W, S, CP>;
    set_smem(k, smem);
    k<<<grid, W * 32, smem, st>>>(reinterpret_cast<const float*>(z.data_ptr()), reinterpret_cast<const float*>(wf.data_ptr()), reinterpret_cast<const float*>(db.data_ptr()),
                                  reinterpret_cast<float*>(dz.data_ptr()), reinterpret_cast<float*>(dwp.data_ptr()), R, (float)eps);
  }
}
}  // namespace

// out [N, R] = W' LN0(z)^T, z [R, C], wf [N, C] (C = 128 or 256), N a multiple of 16 (at most 64)
void hoist_fwd(at::Tensor z, at::Tensor wf, at::Tensor out, double eps) {
  int64_t R, N, C;
  check_rows(z, wf, R, N, C);
  TORCH_CHECK(N <= 64 && out.is_contiguous() && out.scalar_type() == z.scalar_type() && out.size(0) == N && out.size(1) == R, "hoist_fwd: out [N, R], N <= 64");
  const at::cuda::CUDAGuard guard(z.device());
  const int nb = (int)(N / 16);
  if (C == 128) fwd_launch<128>(z, wf, out, R, nb, eps);
  else fwd_launch<256>(z, wf, out, R, nb, eps);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// z [R, C], wf [16, C], db [16, R] (all one dtype) -> dz like z and dwp [grid, 16, C] fp32 (the caller sums dwp over the grid); C = 128 or 256
void hoist_bwd1(at::Tensor z, at::Tensor wf, at::Tensor db, at::Tensor dz, at::Tensor dwp, double eps) {
  int64_t R, N, C;
  check_rows(z, wf, R, N, C);
  TORCH_CHECK(N == 16, "hoist_bwd1: 16 weight rows");
  TORCH_CHECK(db.scalar_type() == z.scalar_type() && db.is_contiguous() && db.size(0) == 16 && db.size(1) == R && dz.scalar_type() == z.scalar_type() && dz.is_contiguous() && dz.sizes() == z.sizes(),
              "hoist_bwd1: db [16, R], dz like z");
  const int grid = num_sms();
  TORCH_CHECK(dwp.scalar_type() == at::kFloat && dwp.is_contiguous() && dwp.numel() == (int64_t)grid * 16 * C, "hoist_bwd1: dwp [", grid, ", 16, ", C, "] fp32");
  const at::cuda::CUDAGuard guard(z.device());
  if (C == 128) bwd_launch<128>(z, wf, db, dz, dwp, R, eps);
  else bwd_launch<256>(z, wf, db, dz, dwp, R, eps);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
int64_t hoist_pass_grid() { return num_sms(); }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("hoist_fwd", &hoist_fwd);
  m.def("hoist_bwd1", &hoist_bwd1);
  m.def("hoist_pass_grid", &hoist_pass_grid);
}
