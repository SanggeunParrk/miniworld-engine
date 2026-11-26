// msg_bwd_fused_sm80.cuh -- the whole backward of the fused hidden-message reduction on A100 (sm_80), weight and bias gradients included:
//
//   per 16-row tile (rows of one group), replaying the forward:   a = bf16(gelu(P))   projected = bf16(a W^T + bias)                            (recomputed: nothing was saved)
//   gh = g[group, o] mask[n] / scale     dproj = bf16(gh gelu'(projected))
//   dX = dproj W   dP = bf16(dX gelu'(P))                                                                                                        (the gradient of the preactivation)
//   dW = sum over rows dproj^T a     db = sum over rows dproj                                                                                    (fp32, a fixed summation order)
//
// msg_bwd_sm80.cuh writes `a` and `dproj` to HBM and leaves dW to cuBLAS (2 x 200 MB of stores and 2 x 200 MB of reads at 16384 groups, and a 0.28 ms GEMM).  Here the CTA's 8 warps work in STAGES of one 16-row tile
// each: every warp runs the per-tile pipeline of msg_bwd_sm80.cuh (replay GEMM, epilogue, dX, dP) and leaves its `a` tile and its dproj^T tile in shared memory; after a barrier warp w multiplies, for every tile of the
// stage, ITS 16 rows of dproj^T [o][n] with a [n][i] into its 16 x 128 slice of dW -- 16 mma and 9 ldmatrix per tile, the slice kept in 64 fp32 registers across the whole kernel (A = dproj^T is already in the A-fragment
// layout of the tile; B = a through ldmatrix.trans) -- and with a B of ones into the same rows of db (one more mma; every column of its result is the row sum).  Per-CTA dW slices go to one [ctas][128][128] buffer and
// a fixed-order reduction kernel sums them: no atomics, bit-reproducible.
//
// What was measured to pay (A100, 16384 groups, one process, us; v1 = the barrier-per-stage kernel as first written, 986): the tensor-bound dW phase and the FP32-bound GELU of the next tile do not compete for the same
// pipe when the two warps of a scheduler are staggered -- after the stage barrier warps 0-3 run their dW slice and then the first phase (GELU of P -> the B fragments `bfr`) of their next tile, warps 4-7 the other way
// round (MP_FUSED_STAGGER, -4.5 %; the `a` tile is written to shared memory only after the second barrier, from the registers `bfr` that stay live); the 16 group-gradient values of a lane are read when the tile starts,
// not at their use (MP_FUSED_GPF, -4.4 %: the L2 latency stalled the first mma); dP leaves through a 4 KB shared staging tile as whole 128-byte row segments instead of 4-byte stores of 8 rows x 16 B (MP_FUSED_DPSTAGE,
// -9 %: the load-store queue throttled); the bias gradient as a ones-mma instead of per-lane sums and two shuffle rounds (-0..4 %, 16 fewer live registers).  Not kept: gelu and gelu' of P from one evaluation with gelu'
// stashed as fp16 in the P tile (-1..3 %, one more rounding in dP).
#pragma once
#include "mpnn_common.cuh"

namespace mp80 {

struct MsgBwdFParams {
  const bf* p;            // [tiles * 16][128]
  const bf* w;            // [128][128]
  const bf* bias;         // [128]
  const float* mask;      // [tiles * 16]
  const float* gred;      // [groups][128] fp32: the gradient of `reduced`
  bf* dp;                 // [tiles * 16][128]: the gradient of P
  float* dw_part;         // [ctas][128][128]
  float* db_part;         // [ctas][128]
  int64_t tiles;          // groups * 3
  float inv_scale;
};

#ifndef MP_FUSED_STAGGER
#define MP_FUSED_STAGGER 1       // 1: warps 0-3 run dW then the next tile's GELU, warps 4-7 the reverse (the tensor pipe and the FP32 pipe of one scheduler work at the same time)
#endif
#ifndef MP_FUSED_GPF
#define MP_FUSED_GPF 1           // 1: the 16 group-gradient values of a lane are loaded when the tile starts (not at their use)
#endif
#ifndef MP_FUSED_DPSTAGE
#define MP_FUSED_DPSTAGE 1       // 1: dP leaves through a 4 KB shared staging tile per warp as 128-byte row segments
#endif

struct MsgBwdFCfg {
  static constexpr int NW = 8, NTHR = NW * 32, MINB = 1;
  static constexpr int W_BYTES = 128 * 256, TILE = 16 * 256, DPT = 128 * 32, ATILE = 16 * 256, MASKB = 128, STG = MP_FUSED_DPSTAGE ? 16 * 256 : 0;
  static constexpr int OFF_DPT = TILE, OFF_A = TILE + DPT, OFF_MASK = TILE + DPT + ATILE, OFF_STG = OFF_MASK + MASKB, WSTRIDE = OFF_STG + STG;   // per warp: P tile | dproj^T [128 o][16 n] | a tile [16 n][128 i] | 16 mask floats | the dP staging tile
  static constexpr int SMEM = W_BYTES + 512 + NW * WSTRIDE;                                                              // + the bias (256 B, padded)
};

// the dproj^T tile: 128 rows (o) of 32 B (16 n), 16-B granule jn of row o stored at granule jn ^ ((o >> 2) & 1)
DEVI uint32_t dpt_swz_f(uint32_t o, uint32_t jn) { return o * 32u + ((jn ^ ((o >> 2) & 1u)) << 4); }

__global__ void __launch_bounds__(MsgBwdFCfg::NTHR, MsgBwdFCfg::MINB) msg_bwd_fused_kernel(const MsgBwdFParams p) {
  using C = MsgBwdFCfg;
  extern __shared__ __align__(128) unsigned char smem_raw[];
  const uint32_t sb = smem_u32(smem_raw);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g8 = lane >> 2, q4 = lane & 3;

  for (int i = tid; i < 2048; i += C::NTHR) {
    const int r = i >> 4, g = i & 15;
    cp_async16(sb + sw256(r, g), p.w + r * 128 + g * 8);
  }
  if (tid < 16) cp_async16(sb + C::W_BYTES + tid * 16, p.bias + tid * 8);
  cp_async_commit();

  const int64_t nstages = (p.tiles + C::NW - 1) / C::NW;
  const int64_t cta = blockIdx.x, nctas = gridDim.x;
  const uint32_t wb = sb + C::W_BYTES + 512 + warp * C::WSTRIDE;      // this warp's region
  const uint32_t tb = wb, dpt = wb + C::OFF_DPT, atb = wb + C::OFF_A, mkb = wb + C::OFF_MASK, stgb = wb + C::OFF_STG;
  const bf* sbias = reinterpret_cast<const bf*>(smem_raw + C::W_BYTES);

  const int mi = lane >> 3, x7 = lane & 7;
  const uint32_t abase = sb + ((mi & 1) * 8 + x7) * 256;                // W as the A operand of the replay GEMM (non-trans)
  const uint32_t bbase = tb + ((mi >> 1) * 8 + x7) * 256;               // the P tile as B fragments / raw fragments
  const int ga = mi >> 1, gb = mi & 1;

  auto issue = [&](int64_t tile) {                                      // the tile's 16 rows of P and mask (callers check that the tile exists)
    const int64_t r0 = tile * 16;
    const bf* src = p.p + r0 * 128;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int c = lane + 32 * i;
      cp_async16(tb + sw256(c >> 4, c & 15), src + c * 8);
    }
    if (lane < 4) cp_async16(mkb + lane * 16, p.mask + r0 + lane * 4);
  };

  cp_async_wait<0>();
  __syncthreads();
  if (cta < nstages && cta * C::NW + warp < p.tiles) issue(cta * C::NW + warp);
  cp_async_commit();

  float accw[16][4];                                                    // this warp's dW slice: rows o = 16 warp + g8 (+ 8), columns i = 8 j + 2 q4 (+ 1)
#pragma unroll
  for (int j = 0; j < 16; ++j) { accw[j][0] = accw[j][1] = accw[j][2] = accw[j][3] = 0.f; }
  float accb[4] = {0.f, 0.f, 0.f, 0.f};                                 // the bias gradient of the same rows: dproj^T x (a column of ones) on the tensor core, the same fp32 accumulation as dW

  uint32_t bfr[8][4];                                                   // a = bf16(gelu(P)) of the current tile as B fragments
  float mks[2][2];                                                      // mask * inv_scale of the current tile: columns n = 2 q4 (+1) and 8 + 2 q4 (+1)

  // phase A: the tile has landed -> its mask and the GELU of P
  auto phase_a = [&]() {
    cp_async_wait<0>();
    __syncwarp();
    const uint2 m0 = lds64(mkb + (2 * q4) * 4), m1 = lds64(mkb + (8 + 2 * q4) * 4);
    mks[0][0] = __uint_as_float(m0.x) * p.inv_scale; mks[0][1] = __uint_as_float(m0.y) * p.inv_scale;
    mks[1][0] = __uint_as_float(m1.x) * p.inv_scale; mks[1][1] = __uint_as_float(m1.y) * p.inv_scale;
    uint32_t praw[8][4];
#pragma unroll
    for (int s = 0; s < 8; ++s) ldsm_x4(praw[s], bbase + ((((2 * s + gb) ^ x7)) << 4));
#pragma unroll
    for (int s = 0; s < 8; ++s)
#pragma unroll
      for (int i = 0; i < 4; ++i) bfr[s][i] = pack_bf16(gelu_f(bf16lo(praw[s][i])), gelu_f(bf16hi(praw[s][i])));
  };

  // the dW and db slices of this warp over the tiles of the stage
  auto phase_w = [&](int nact) {
#pragma unroll 1
    for (int t = 0; t < nact; ++t) {
      const uint32_t twb = sb + C::W_BYTES + 512 + t * C::WSTRIDE;
      uint32_t a[4];
      ldsm_x4(a, twb + C::OFF_DPT + dpt_swz_f(16 * warp + (mi & 1) * 8 + x7, mi >> 1));
#pragma unroll
      for (int jp = 0; jp < 8; ++jp) {
        uint32_t bw[4];
        ldsm_x4_t(bw, twb + C::OFF_A + sw256(8 * (mi & 1) + x7, 2 * jp + (mi >> 1)));
        mma16816(accw[2 * jp], a, bw[0], bw[1]);
        mma16816(accw[2 * jp + 1], a, bw[2], bw[3]);
      }
      mma16816(accb, a, 0x3F803F80u, 0x3F803F80u);                      // B = ones (bf16 1.0 pairs): every column of the result is the row sum of dproj^T = the bias gradient
    }
  };

  if (cta < nstages && cta * C::NW + warp < p.tiles) phase_a();

  for (int64_t st = cta; st < nstages; st += nctas) {
    const int64_t tile = st * C::NW + warp;
    const int64_t rem = p.tiles - st * C::NW;
    const int nact = rem < C::NW ? (int)rem : C::NW;                    // tiles of this stage (the last stage may be short)
    if (warp < nact) {
      const int64_t r0 = tile * 16;
      const float* gp = p.gred + (tile / 3) * 128;                      // the group's gradient row g[16 m + g8 + 8 h]
#if MP_FUSED_GPF
      float gg16[8][2];
#pragma unroll
      for (int m = 0; m < 8; ++m)
#pragma unroll
        for (int h = 0; h < 2; ++h) gg16[m][h] = ldg_f32(gp + 16 * m + g8 + 8 * h);
#endif
      // a -> its shared tile [16 n][128 i] (for dW)
#pragma unroll
      for (int s = 0; s < 8; ++s)
#pragma unroll
        for (int i = 0; i < 4; ++i) sts32(atb + sw256(g8 + 8 * (i >> 1), 2 * s + (i & 1)) + 4 * q4, bfr[s][i]);

      // ---- replay GEMM (transposed) and its epilogue: dproj^T[o][n] -> the shared tile
#pragma unroll
      for (int mp = 0; mp < 4; ++mp) {                                  // two output m-tiles at a time: four independent accumulator chains per k step
        float acc[2][2][4];
#pragma unroll
        for (int u = 0; u < 2; ++u)
#pragma unroll
          for (int j = 0; j < 2; ++j) { acc[u][j][0] = acc[u][j][1] = acc[u][j][2] = acc[u][j][3] = 0.f; }
#pragma unroll
        for (int s = 0; s < 8; ++s) {
          uint32_t a0[4], a1[4];
          ldsm_x4(a0, abase + (2 * mp) * 4096 + (((2 * s + ga) ^ x7) << 4));
          ldsm_x4(a1, abase + (2 * mp + 1) * 4096 + (((2 * s + ga) ^ x7) << 4));
          mma16816(acc[0][0], a0, bfr[s][0], bfr[s][1]);
          mma16816(acc[1][0], a1, bfr[s][0], bfr[s][1]);
          mma16816(acc[0][1], a0, bfr[s][2], bfr[s][3]);
          mma16816(acc[1][1], a1, bfr[s][2], bfr[s][3]);
        }
#pragma unroll
        for (int u = 0; u < 2; ++u) {
          const int m = 2 * mp + u;
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            const int o = 16 * m + g8 + 8 * h;
            const float b = __bfloat162float(sbias[o]);
#if MP_FUSED_GPF
            const float gg = gg16[m][h];
#else
            const float gg = __ldg(gp + o);
#endif
#pragma unroll
            for (int j = 0; j < 2; ++j) {
              const uint32_t pr = pack_bf16(acc[u][j][2 * h] + b, acc[u][j][2 * h + 1] + b);               // projected
              const uint32_t dp = pack_bf16(gg * mks[j][0] * gelu_grad_f(bf16lo(pr)), gg * mks[j][1] * gelu_grad_f(bf16hi(pr)));
              sts32(dpt + dpt_swz_f(o, j) + 4 * q4, dp);
            }
          }
        }
      }
      __syncwarp();

      // ---- dX = dproj W (m = n, k = o, N = i), two halves of 64 input channels; dP = bf16(dX gelu'(P))
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        float accx[8][4];
#pragma unroll
        for (int j = 0; j < 8; ++j) { accx[j][0] = accx[j][1] = accx[j][2] = accx[j][3] = 0.f; }
#pragma unroll
        for (int s = 0; s < 8; ++s) {
          uint32_t ad[4];
          ldsm_x4_t(ad, dpt + dpt_swz_f(16 * s + 8 * (mi >> 1) + x7, mi & 1));
#pragma unroll
          for (int jp = 0; jp < 4; ++jp) {
            uint32_t bw[4];
            ldsm_x4_t(bw, sb + sw256(16 * s + 8 * (mi & 1) + x7, 8 * half + 2 * jp + (mi >> 1)));
            mma16816(accx[2 * jp], ad, bw[0], bw[1]);
            mma16816(accx[2 * jp + 1], ad, bw[2], bw[3]);
          }
        }
        uint32_t praw[4][4];                                            // the raw fragments of this half's 4 channel steps (the tile stays in shared memory until here)
#pragma unroll
        for (int ss = 0; ss < 4; ++ss) ldsm_x4(praw[ss], bbase + ((((2 * (4 * half + ss) + gb) ^ x7)) << 4));
        if (half == 1) {                                                // the P tile is dead: the next stage's tile may land in its place
          __syncwarp();
          const int64_t next = (st + nctas) * C::NW + warp;
          if (st + nctas < nstages && next < p.tiles) issue(next);
          cp_async_commit();
        }
#pragma unroll
        for (int jj = 0; jj < 8; ++jj) {
          const int j = 8 * half + jj;
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            const uint32_t pr = praw[jj >> 1][2 * h + (jj & 1)];
            const float d0 = accx[jj][2 * h] * gelu_grad_f(bf16lo(pr)), d1 = accx[jj][2 * h + 1] * gelu_grad_f(bf16hi(pr));
#if MP_FUSED_DPSTAGE
            sts32(stgb + sw256(g8 + 8 * h, j) + 4 * q4, pack_bf16(d0, d1));
#else
            stg32(p.dp + (r0 + g8 + 8 * h) * 128 + 8 * j + 2 * q4, pack_bf16(d0, d1));
#endif
          }
        }
#if MP_FUSED_DPSTAGE
        __syncwarp();                                                   // this half of dP is in the staging tile: 16 rows x 8 granules of 16 B -> four 16-byte stores per lane, 128-byte row segments
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const int c = lane + 32 * i, row = c >> 3, gran = 8 * half + (c & 7);
          stg128(p.dp + (r0 + row) * 128 + gran * 8, lds128(stgb + sw256(row, gran)));
        }
        __syncwarp();
#endif
      }
    }
    __syncthreads();                                                    // the stage's `a` and dproj^T tiles are complete

    // ---- the dW / db slice of this stage and the first phase of the next tile
    const int64_t nst = st + nctas;
    const bool has_next = nst < nstages && nst * C::NW + warp < p.tiles;
#if MP_FUSED_STAGGER
#pragma unroll 1
    for (int ph = 0; ph < 2; ++ph) {
      if ((ph == 0) == (warp < 4)) phase_w(nact);
      else if (has_next) phase_a();
    }
#else
    phase_w(nact);
    if (has_next) phase_a();
#endif
    __syncthreads();                                                    // every slice has read the tiles: they may be rewritten
  }

  // ---- the slices of dW and the bias gradient leave for the reductions
  float* dwp = p.dw_part + (size_t)blockIdx.x * 128 * 128;
#pragma unroll
  for (int j = 0; j < 16; ++j)
#pragma unroll
    for (int h = 0; h < 2; ++h)
      *reinterpret_cast<float2*>(dwp + (16 * warp + g8 + 8 * h) * 128 + 8 * j + 2 * q4) = make_float2(accw[j][2 * h], accw[j][2 * h + 1]);
  cp_async_wait<0>();
  if (q4 == 0) {                                                        // every column of accb holds the row sum: lane (g8, 0) writes rows 16 warp + g8 and + 8
    p.db_part[(size_t)blockIdx.x * 128 + 16 * warp + g8] = accb[0];
    p.db_part[(size_t)blockIdx.x * 128 + 16 * warp + g8 + 8] = accb[2];
  }
}

// dW [128 * 128] = the sum of the CTAs' slices in CTA order
__global__ void dw_reduce_kernel(const float* __restrict__ part, float* __restrict__ out, int nparts) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= 128 * 128) return;
  float s = 0.f;
  for (int c = 0; c < nparts; ++c) s += part[(size_t)c * (128 * 128) + i];
  out[i] = s;
}

}  // namespace mp80
