// b1_sm80.cuh -- the backward of the front (f1_sm80.cuh), A100 / sm_80: the input gradient of the five projections and of the LayerNorm, plus the residual,
// in one pass over the pair tensor.
//
//   dxn = bf16( [dq | dk | dv | dg | db] . [Wq; Wk; Wv; Wg; Wb] )       the LayerNorm output's gradient (K = 516, fp32 accumulation)
//   dxh = dxn gamma;  xh = (x - mean) rstd
//   dx  = bf16( rstd (dxh - mean(dxh) - xh mean(dxh xh)) )                the LayerNorm backward, statistics over the 128 channels
//   dpair = bf16(dout + dx)                                              the residual's gradient joins
//
// The weights' and the LayerNorm affine's gradients are NOT computed here: with G = D^T [xh | 1] (one GEMM over the tokens, D = [dq | dk | dv | dg | db]),
// dW = G diag(gamma) + s beta^T and, since dxn = D W, dgamma_c = sum_o W[o][c] G[o][c] and dbeta_c = sum_o W[o][c] s[o] (s = the column sums of D: the
// ones column of [xh | 1]) -- the module-side code does that algebra on 516 x 128 matrices.
//
// Layout of the front: a persistent CTA per SM with the weights resident in shared memory, a warp walks tiles of 16 tokens (one m16 tile) on its own, the
// tokens go straight from global memory into the A fragments (k order permuted so a thread holds 8 consecutive channels: 16-byte loads), the packed weight
// rows follow f1_channel so the accumulators hold 8 consecutive output channels per thread and the LayerNorm statistics are a quad shuffle away.  The
// contraction is four blocks of 128 (the q, k, v, g gradients: Wp^T [c][o] resident, 4 x 32 KiB) and the bias heads (K = 4, padded to one k step).
// ``x``, ``dout`` and ``dpair`` are in the module's layout (the ending node reads and writes the transposed positions); D, the statistics and db are in
// the starting frame.
#pragma once
#include "f3_sm80.cuh"

namespace a100 {

struct B1Params {
  const __nv_bfloat16* d;       // [T][ldd] dq | dk | dv | dg (columns 0 .. 511)
  const __nv_bfloat16* db;      // [Z][4][L][L] the pair bias' gradient planes
  const __nv_bfloat16* x;       // [T][128] the pair tensor, in the module's layout
  const float* lnst;            // [T][2] (mean, rstd)
  const __nv_bfloat16* dout;    // [T][128] the output's gradient, in the module's layout
  __nv_bfloat16* dpair;         // [T][128] the pair tensor's gradient, in the module's layout
  const __nv_bfloat16* wt;      // [4][128][128] W_p^T: wt[p][c][o] = W_p[o][c], p = q, k, v, g
  const __nv_bfloat16* wbt;     // [128][4]: wbt[c][h] = Wb[h][c]
  const float* gamma;           // [128]
  long long ldd;
  unsigned ntile;               // T / 16
  int L, transposed;
};

template <int NW_>
struct B1Cfg {
  static constexpr int NW = NW_, NTHR = NW_ * 32;
  static constexpr int SMEM = 4 * 128 * 256 + 128 * 8 + 128 * 4;       // 4 x Wp^T (rows of 256 B) + Wb^T (8 B rows) + gamma
};

template <class G>
__global__ void __launch_bounds__(G::NTHR, 1) b1_kernel(const B1Params p) {
  extern __shared__ __align__(128) unsigned char smem_raw[];
  const uint32_t sb = smem_u32(smem_raw);
  unsigned char* wbt_s = smem_raw + 4 * 128 * 256;
  float* gam = reinterpret_cast<float*>(smem_raw + 4 * 128 * 256 + 128 * 8);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g8 = lane >> 2, q4 = lane & 3;

  // ---- the weights into shared memory, once per CTA: packed row s of a block holds the input channel c = f1_channel(s) (its column of W_p, as a row over o)
  for (int i = tid; i < 4 * 128 * 16; i += G::NTHR) {
    const int blk = i >> 11, row = (i >> 4) & 127, ch = i & 15;
    cp_async16(sb + blk * 32768 + f1_woff(row, ch), p.wt + ((size_t)blk * 128 + f1_channel(row)) * 128 + ch * 8);
  }
  cp_async_commit();
  for (int i = tid; i < 128 * 4; i += G::NTHR) reinterpret_cast<__nv_bfloat16*>(wbt_s)[i] = p.wbt[(size_t)f1_channel(i >> 2) * 4 + (i & 3)];
  for (int i = tid; i < 128; i += G::NTHR) gam[i] = p.gamma[i];
  cp_async_wait<0>();
  __syncthreads();

  const unsigned L = (unsigned)p.L, LL = L * L;
  uint32_t wb[4];
#pragma unroll
  for (int kb = 0; kb < 4; ++kb) wb[kb] = sb + f1_woff(g8, 4 * kb + q4);

  for (unsigned tile = blockIdx.x * G::NW + warp; tile < p.ntile; tile += gridDim.x * G::NW) {
    // 16 tokens of one pair row (L is a multiple of 128): problem z, row arow, columns b0 .. b0 + 15
    const unsigned t0 = tile * 16, z = t0 / LL, rem0 = t0 - z * LL, arow = rem0 / L, b0 = rem0 - arow * L;
    size_t drow[2];                                              // the token's row in the module's layout (x, dout, dpair)
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
      const unsigned r = g8 + 8 * hh;
      drow[hh] = p.transposed ? (size_t)(z * LL + (b0 + r) * L + arow) : (size_t)(t0 + r);
    }

    float acc[16][4];
#pragma unroll
    for (int nt = 0; nt < 16; ++nt) acc[nt][0] = acc[nt][1] = acc[nt][2] = acc[nt][3] = 0.f;

    // ---- dxn: the four gradient blocks (A fragments from global memory, the next block prefetched), then the bias heads
    uint4 av[2][4];
#pragma unroll
    for (int hh = 0; hh < 2; ++hh)
#pragma unroll
      for (int kb = 0; kb < 4; ++kb) av[hh][kb] = ldg128(p.d + ((size_t)t0 + g8 + 8 * hh) * p.ldd + 32 * kb + 8 * q4);
    uint4 xv[2][4], ov[2][4];                                    // x and dout rows for the epilogue, loaded during the last block
#pragma unroll
    for (int blk = 0; blk < 4; ++blk) {
      uint32_t a[8][4];
#pragma unroll
      for (int kb = 0; kb < 4; ++kb)
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
          const uint4 u = av[hh][kb];
          a[2 * kb][hh] = u.x;      a[2 * kb][2 + hh] = u.y;
          a[2 * kb + 1][hh] = u.z;  a[2 * kb + 1][2 + hh] = u.w;
        }
      if (blk < 3) {
#pragma unroll
        for (int hh = 0; hh < 2; ++hh)
#pragma unroll
          for (int kb = 0; kb < 4; ++kb) av[hh][kb] = ldg128(p.d + ((size_t)t0 + g8 + 8 * hh) * p.ldd + 128 * (blk + 1) + 32 * kb + 8 * q4);
      } else {
#pragma unroll
        for (int hh = 0; hh < 2; ++hh)
#pragma unroll
          for (int kb = 0; kb < 4; ++kb) {
            xv[hh][kb] = ldg128(p.x + drow[hh] * 128 + 32 * kb + 8 * q4);
            ov[hh][kb] = ldg128(p.dout + drow[hh] * 128 + 32 * kb + 8 * q4);
          }
      }
      uint4 bn = lds128_ro(wb[0] + blk * 32768u);
#pragma unroll
      for (int kb = 0; kb < 4; ++kb)
#pragma unroll
        for (int nt = 0; nt < 16; ++nt) {
          const uint4 bc = bn;
          if (kb * 16 + nt < 63) bn = lds128_ro(wb[(kb * 16 + nt + 1) >> 4] + blk * 32768u + ((kb * 16 + nt + 1) & 15) * 8 * 256);
          mma16816(acc[nt], a[2 * kb], bc.x, bc.y);
          mma16816(acc[nt], a[2 * kb + 1], bc.z, bc.w);
        }
    }
    {                                                            // the bias heads: one k step (k = head, 4 of 16 real: lanes q4 < 2 hold heads 2 q4, 2 q4 + 1)
      uint32_t adb[4] = {0u, 0u, 0u, 0u};
      if (q4 < 2) {
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
          const size_t at = (size_t)(z * 4 + 2 * q4) * LL + rem0 + g8 + 8 * hh;
          const uint32_t lo = __bfloat16_as_ushort(p.db[at]), hi = __bfloat16_as_ushort(p.db[at + LL]);
          adb[hh] = lo | (hi << 16);
        }
      }
#pragma unroll
      for (int nt = 0; nt < 16; ++nt) {
        const uint32_t b0 = q4 < 2 ? lds32(sb + 4 * 128 * 256 + (nt * 8 + g8) * 8 + 4 * q4) : 0u;
        mma16816(acc[nt], adb, b0, 0u);
      }
    }

    // ---- the LayerNorm backward on the accumulators (a thread holds 32 of the 128 channels of its two rows, a quad the whole row)
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
      const size_t tok = (size_t)t0 + g8 + 8 * hh;
      const float2 st = *reinterpret_cast<const float2*>(p.lnst + 2 * tok);
      const float mean = st.x, rstd = st.y;
      float s1 = 0.f, s2 = 0.f;
#pragma unroll
      for (int og = 0; og < 4; ++og) {
        const uint4 xu = xv[hh][og];
        const uint32_t xw[4] = {xu.x, xu.y, xu.z, xu.w};
        const float4 g0 = *reinterpret_cast<const float4*>(gam + 32 * og + 8 * q4), g1 = *reinterpret_cast<const float4*>(gam + 32 * og + 8 * q4 + 4);
        const float gg[8] = {g0.x, g0.y, g0.z, g0.w, g1.x, g1.y, g1.z, g1.w};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const float x0 = bf16lo(xw[j]), x1 = bf16hi(xw[j]);
          const float d0 = round_bf16f(acc[4 * og + j][2 * hh]) * gg[2 * j], d1 = round_bf16f(acc[4 * og + j][2 * hh + 1]) * gg[2 * j + 1];
          s1 += d0 + d1;
          s2 = fmaf(d0, (x0 - mean) * rstd, fmaf(d1, (x1 - mean) * rstd, s2));
        }
      }
      const float m1 = quad_sum(s1) * (1.f / 128.f), m2 = quad_sum(s2) * (1.f / 128.f);
#pragma unroll
      for (int og = 0; og < 4; ++og) {
        const uint4 xu = xv[hh][og], ou = ov[hh][og];
        const uint32_t xw[4] = {xu.x, xu.y, xu.z, xu.w}, ow[4] = {ou.x, ou.y, ou.z, ou.w};
        const float4 g0 = *reinterpret_cast<const float4*>(gam + 32 * og + 8 * q4), g1 = *reinterpret_cast<const float4*>(gam + 32 * og + 8 * q4 + 4);
        const float gg[8] = {g0.x, g0.y, g0.z, g0.w, g1.x, g1.y, g1.z, g1.w};
        uint32_t r[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const float xh0 = (bf16lo(xw[j]) - mean) * rstd, xh1 = (bf16hi(xw[j]) - mean) * rstd;
          const float d0 = round_bf16f(acc[4 * og + j][2 * hh]) * gg[2 * j], d1 = round_bf16f(acc[4 * og + j][2 * hh + 1]) * gg[2 * j + 1];
          r[j] = add_bf16x2(ow[j], pack_bf16(rstd * (d0 - m1 - xh0 * m2), rstd * (d1 - m1 - xh1 * m2)));
        }
        stg128(p.dpair + drow[hh] * 128 + 32 * og + 8 * q4, make_uint4(r[0], r[1], r[2], r[3]));
      }
    }
  }
}

}  // namespace a100
