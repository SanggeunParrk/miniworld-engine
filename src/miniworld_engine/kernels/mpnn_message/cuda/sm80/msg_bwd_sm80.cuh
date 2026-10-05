// msg_bwd_sm80.cuh -- the backward of the fused hidden-message reduction on A100 (sm_80): everything of it but the weight-gradient GEMM.
//
//   per 16-row tile (rows of one group), replaying the forward:   a = bf16(gelu(P))   projected = bf16(a W^T + bias)                            (recomputed: nothing was saved)
//   gh = bf16(g[group, o] mask[n] / scale)     dproj = bf16(float(gh) gelu'(projected))     db += dproj
//   dX = bf16(dproj W)   dP = bf16(float(dX) gelu'(P))                                                                                          (the gradient of the preactivation)
//
// and it writes dP, a and dproj ([rows, 128] bf16) so that dW = dproj^T a is ONE cuBLAS GEMM over the rows (the integration chunks the rows to bound the two temporaries).  The bias gradient is summed
// in registers (fp32, a fixed order), per CTA, and written as one partial row per CTA: a small deterministic reduction finishes it (no atomics anywhere).
//
// The two GEMMs use opposite orientations so no fragment ever crosses lanes through shuffles: the replay GEMM is run transposed (A = W from shared memory, B = the GELU'd activation tile in registers: C^T [o][n],
// the layout of the forward kernel), its epilogue (bias, gelu', the mask and group gradient: per o and per n) writes dproj^T [o][n] to a small shared tile (STS.32 pairs along n), and ldmatrix.trans reads it back
// as the A fragments of dX = dproj W (m = n, k = o; B = W through ldmatrix.trans), the same tile stored as the dproj output.  gelu'(P) needs the raw P: the 32 registers of the tile's fragments stay live.
// Register budget (it spilled at 168 registers): the group gradient is re-read from L1 where it is used, the bias gradient is a reduce-scatter over the quad into 4 persistent registers, and dX is computed in
// two halves of 64 input channels (32 accumulators live at a time instead of 64).
#pragma once
#include "mpnn_common.cuh"

namespace mp80 {

struct MsgBwdParams {
  const bf* p;            // [groups * 48][128]
  const bf* w;            // [128][128]
  const bf* bias;         // [128]
  const float* mask;      // [groups * 48]
  const float* gred;      // [groups][128] fp32: the gradient of `reduced`
  bf* dp;                 // [groups * 48][128]: the gradient of P
  bf* a_out;              // [groups * 48][128]: bf16(gelu(P))   (for dW)
  bf* dproj_out;          // [groups * 48][128]: the gradient of the projection (for dW)
  float* db_part;         // [ctas][128]: the bias gradient, one partial row per CTA
  int64_t groups;
  float inv_scale;
};

#ifndef MP_BWD_STAGE
#define MP_BWD_STAGE 0           // 1: the three outputs (a, dproj, dP) leave through a shared staging tile as 16-byte stores (a store from the fragment layout is 4 bytes per lane: 8 rows x 16 B per instruction).
                                 // Measured on the A100: 1383 us instead of 1314 us at 16384 groups (the extra shared-memory traffic and the 10 instead of 12 warps cost more than the wider stores save): off.
#endif
#ifndef MP_BWD_NW
#define MP_BWD_NW (MP_BWD_STAGE ? 10 : 12)
#endif
#ifndef MP_BWD_RELOAD
#define MP_BWD_RELOAD 1          // 1: the raw P fragments are not kept in registers through the replay GEMM: the tile stays in shared memory and is read again for the dX epilogue
#endif

struct MsgBwdCfg {
  static constexpr int NW = MP_BWD_NW, NTHR = NW * 32, MINB = 1;
  static constexpr int W_BYTES = 128 * 256, TILE = 16 * 256, DPT = 128 * 32, STG = MP_BWD_STAGE ? 16 * 256 : 0;
  static constexpr int WSTRIDE = TILE + DPT + 128 + STG;                                                      // P tile | dproj^T [128 o][16 n] | 16 mask floats (padded) | the staging tile
  static constexpr int SMEM = W_BYTES + 512 + NW * WSTRIDE;                                                    // + the bias (256 B, padded)
};

#define STG_A(p, v) stg32(p, v)
#define STG_D(p, v) stg32(p, v)
#define STG_P(p, v) stg32(p, v)

// the dproj^T tile: 128 rows (o) of 32 B (16 n), 16-B granule jn of row o stored at granule jn ^ ((o >> 2) & 1)
DEVI uint32_t dpt_swz(uint32_t o, uint32_t jn) { return o * 32u + ((jn ^ ((o >> 2) & 1u)) << 4); }

__global__ void __launch_bounds__(MsgBwdCfg::NTHR, MsgBwdCfg::MINB) msg_bwd_kernel(const MsgBwdParams p) {
  using C = MsgBwdCfg;
  extern __shared__ __align__(128) unsigned char smem_raw[];
  const uint32_t sb = smem_u32(smem_raw);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g8 = lane >> 2, q4 = lane & 3;

  for (int i = tid; i < 2048; i += C::NTHR) {
    const int r = i >> 4, g = i & 15;
    cp_async16(sb + sw256(r, g), p.w + r * 128 + g * 8);
  }
  if (tid < 16) cp_async16(sb + C::W_BYTES + tid * 16, p.bias + tid * 8);
  cp_async_commit();

  const int64_t total_warps = (int64_t)gridDim.x * C::NW;
  const int64_t gw = (int64_t)blockIdx.x * C::NW + warp;
  const int64_t ng = gw < p.groups ? (p.groups - gw + total_warps - 1) / total_warps : 0;
  const int64_t ntiles = ng * 3;
  const uint32_t tb = sb + C::W_BYTES + 512 + warp * C::WSTRIDE;      // the P tile; then the dproj^T tile at tb + TILE; then the mask
  const uint32_t dpt = tb + C::TILE, mkb = dpt + C::DPT, stgb = mkb + 128;
  const bf* sbias = reinterpret_cast<const bf*>(smem_raw + C::W_BYTES);
  // the staging tile [16 rows][256 B] (swizzled like the P tile) -> `chunks` 16-byte stores per lane of rows 0 .. 15, columns gran0 .. gran0 + width - 1 of 16-byte granules (width 16: whole rows, 8 per lane; 8: half rows)
  auto flush = [&](bf* dst, int gran0, int width) {
    const int per_row = width, n = 16 * width / 32;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      if (i < n) {
        const int c = lane + 32 * i, row = c / per_row, gran = gran0 + (c % per_row);
        stg128(dst + row * 128 + gran * 8, lds128(stgb + sw256(row, gran)));
      }
    }
  };

  const int mi = lane >> 3, x7 = lane & 7;
  const uint32_t abase = sb + ((mi & 1) * 8 + x7) * 256;                // W as the A operand of the replay GEMM (non-trans)
  const uint32_t bbase = tb + ((mi >> 1) * 8 + x7) * 256;               // the P tile as B fragments / raw fragments
  const int ga = mi >> 1, gb = mi & 1;

  auto tile_row0 = [&](int64_t k) { return ((gw + (k / 3) * total_warps) * 3 + k % 3) * 16; };
  auto issue = [&](int64_t k) {
    const int64_t r0 = tile_row0(k);
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
  if (ntiles > 0) issue(0);
  cp_async_commit();

  // the bias gradient: after a tile's quad reduce-scatter lane q4 holds the sums of indices idx = base .. base + 3 (idx = 2 m + h <-> o = 16 m + g8 + 8 h), base = (q4 >> 1) * 8 + (q4 & 1) * 4
  float dbp[4] = {0.f, 0.f, 0.f, 0.f};
  const bool hi2 = (q4 & 2) != 0, hi1 = (q4 & 1) != 0;
  for (int64_t k = 0; k < ntiles; ++k) {
    const int64_t r0 = tile_row0(k);
    const float* gp = p.gred + (r0 / 48) * 128;                       // the group's gradient row g[16 m + g8 + 8 h]
    cp_async_wait<0>();
    __syncwarp();
#if !MP_BWD_RELOAD
    uint32_t praw[8][4];                                             // the raw preactivation, B-fragment layout (n = g8 + 8 h, channels 16 s + 8 b + 2 q4 (+1)) = praw[s][2 h + b]
#pragma unroll
    for (int s = 0; s < 8; ++s) ldsm_x4(praw[s], bbase + ((((2 * s + gb) ^ x7)) << 4));
#endif
    const uint2 m0 = lds64(mkb + (2 * q4) * 4), m1 = lds64(mkb + (8 + 2 * q4) * 4);
    const float mks[2][2] = {{__uint_as_float(m0.x) * p.inv_scale, __uint_as_float(m0.y) * p.inv_scale},
                             {__uint_as_float(m1.x) * p.inv_scale, __uint_as_float(m1.y) * p.inv_scale}};
    uint32_t bfr[8][4];                                              // a = bf16(gelu(P))
    {
#if MP_BWD_RELOAD
      uint32_t praw[8][4];
#pragma unroll
      for (int s = 0; s < 8; ++s) ldsm_x4(praw[s], bbase + ((((2 * s + gb) ^ x7)) << 4));
#endif
#pragma unroll
      for (int s = 0; s < 8; ++s)
#pragma unroll
        for (int i = 0; i < 4; ++i) bfr[s][i] = pack_bf16(gelu_f(bf16lo(praw[s][i])), gelu_f(bf16hi(praw[s][i])));
    }
#if !MP_BWD_RELOAD
    __syncwarp();
    if (k + 1 < ntiles) issue(k + 1);
    cp_async_commit();
#endif

    float vs[16];                                                    // this tile's bias gradient, per (m, h)
    {
#if MP_BWD_STAGE
#pragma unroll
      for (int s = 0; s < 8; ++s)
#pragma unroll
        for (int i = 0; i < 4; ++i) sts32(stgb + sw256(g8 + 8 * (i >> 1), 2 * s + (i & 1)) + 4 * q4, bfr[s][i]);
      __syncwarp();
      flush(p.a_out + r0 * 128, 0, 16);
      __syncwarp();
#else
#pragma unroll
      for (int s = 0; s < 8; ++s)
#pragma unroll
        for (int i = 0; i < 4; ++i)
          STG_A(p.a_out + (r0 + 8 * (i >> 1) + g8) * 128 + 16 * s + 8 * (i & 1) + 2 * q4, bfr[s][i]);
#endif

      // ---- replay GEMM (transposed) and its epilogue: dproj^T[o][n] -> the shared tile
#pragma unroll
      for (int mp = 0; mp < 4; ++mp) {                                // two output m-tiles at a time: four independent accumulator chains per k step
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
            const float b = __bfloat162float(sbias[o]), gg = __ldg(gp + o);
            float s2 = 0.f;
#pragma unroll
            for (int j = 0; j < 2; ++j) {
              const uint32_t pr = pack_bf16(acc[u][j][2 * h] + b, acc[u][j][2 * h + 1] + b);             // projected
              // g mask / scale stays fp32 into the GELU derivative (the Triton path rounds it to bf16 first: one rounding fewer, and the product is rounded once, to bf16)
              const uint32_t dp = pack_bf16(gg * mks[j][0] * gelu_grad_f(bf16lo(pr)), gg * mks[j][1] * gelu_grad_f(bf16hi(pr)));
              s2 += bf16lo(dp) + bf16hi(dp);
              sts32(dpt + dpt_swz(o, j) + 4 * q4, dp);
            }
            vs[2 * m + h] = s2;
          }
        }
      }
    }
    // reduce-scatter of the 16 per-tile sums over the quad: 12 shuffles, 4 persistent accumulators
    {
      float w8[8];
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const float send = hi2 ? vs[i] : vs[i + 8], keep = hi2 ? vs[i + 8] : vs[i];
        w8[i] = keep + __shfl_xor_sync(0xffffffffu, send, 2);
      }
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const float send = hi1 ? w8[i] : w8[i + 4], keep = hi1 ? w8[i + 4] : w8[i];
        dbp[i] += keep + __shfl_xor_sync(0xffffffffu, send, 1);
      }
    }
    __syncwarp();

    // ---- dX = dproj W (m = n, k = o, N = i), two halves of 64 input channels; the A fragments are the transposed read of the dproj^T tile, which are also dproj's rows for the output
#define MP_STR_(x) #x
#define MP_XSTR_(x) MP_STR_(x)
#define MP_PRAGMA(x) _Pragma(MP_XSTR_(x))
#ifndef MP_BWD_HALF_UNROLL
#define MP_BWD_HALF_UNROLL 2
#endif
  MP_PRAGMA(unroll MP_BWD_HALF_UNROLL)
    for (int half = 0; half < 2; ++half) {
      float accx[8][4];
#pragma unroll
      for (int j = 0; j < 8; ++j) { accx[j][0] = accx[j][1] = accx[j][2] = accx[j][3] = 0.f; }
#pragma unroll
      for (int s = 0; s < 8; ++s) {
        uint32_t ad[4];
        ldsm_x4_t(ad, dpt + dpt_swz(16 * s + 8 * (mi >> 1) + x7, mi & 1));
        if (half == 0) {
#if MP_BWD_STAGE
          sts32(stgb + sw256(g8, 2 * s) + 4 * q4, ad[0]);
          sts32(stgb + sw256(g8 + 8, 2 * s) + 4 * q4, ad[1]);
          sts32(stgb + sw256(g8, 2 * s + 1) + 4 * q4, ad[2]);
          sts32(stgb + sw256(g8 + 8, 2 * s + 1) + 4 * q4, ad[3]);
#else
          STG_D(p.dproj_out + (r0 + g8) * 128 + 16 * s + 2 * q4, ad[0]);
          STG_D(p.dproj_out + (r0 + g8 + 8) * 128 + 16 * s + 2 * q4, ad[1]);
          STG_D(p.dproj_out + (r0 + g8) * 128 + 16 * s + 8 + 2 * q4, ad[2]);
          STG_D(p.dproj_out + (r0 + g8 + 8) * 128 + 16 * s + 8 + 2 * q4, ad[3]);
#endif
        }
#pragma unroll
        for (int jp = 0; jp < 4; ++jp) {
          uint32_t bw[4];
          ldsm_x4_t(bw, sb + sw256(16 * s + 8 * (mi & 1) + x7, 8 * half + 2 * jp + (mi >> 1)));
          mma16816(accx[2 * jp], ad, bw[0], bw[1]);
          mma16816(accx[2 * jp + 1], ad, bw[2], bw[3]);
        }
      }
#if MP_BWD_STAGE
      if (half == 0) {
        __syncwarp();
        flush(p.dproj_out + r0 * 128, 0, 16);
        __syncwarp();
      }
#endif
#if MP_BWD_RELOAD
      uint32_t praw[4][4];                                           // the raw fragments of this half's 4 channel steps s = 4 half .. 4 half + 3 (praw[s - 4 half][2 h + b])
#pragma unroll
      for (int ss = 0; ss < 4; ++ss) ldsm_x4(praw[ss], bbase + ((((2 * (4 * half + ss) + gb) ^ x7)) << 4));
      if (half == 1) {                                               // the tile is dead: the next one may land in its place
        __syncwarp();
        if (k + 1 < ntiles) issue(k + 1);
        cp_async_commit();
      }
#endif
#pragma unroll
      for (int jj = 0; jj < 8; ++jj) {
        const int j = 8 * half + jj;
#pragma unroll
        for (int h = 0; h < 2; ++h) {
#if MP_BWD_RELOAD
          const uint32_t pr = praw[jj >> 1][2 * h + (jj & 1)];
#else
          const uint32_t pr = praw[j >> 1][2 * h + (j & 1)];
#endif
          const float d0 = accx[jj][2 * h] * gelu_grad_f(bf16lo(pr)), d1 = accx[jj][2 * h + 1] * gelu_grad_f(bf16hi(pr));            // dX stays fp32 into the derivative (one rounding, at the store)
#if MP_BWD_STAGE
          sts32(stgb + sw256(g8 + 8 * h, j) + 4 * q4, pack_bf16(d0, d1));
#else
          STG_P(p.dp + (r0 + g8 + 8 * h) * 128 + 8 * j + 2 * q4, pack_bf16(d0, d1));
#endif
        }
      }
#if MP_BWD_STAGE
      __syncwarp();
      flush(p.dp + r0 * 128, 8 * half, 8);
      __syncwarp();
#endif
    }
    __syncwarp();                                                    // the dproj^T tile is rewritten by the next tile's epilogue
  }

  // ---- the bias gradient: the warps of the CTA in warp order
  cp_async_wait<0>();
  __syncthreads();
  float* part = reinterpret_cast<float*>(smem_raw);                  // the weight is dead: [NW][128] floats
  {
    const int base = (q4 >> 1) * 8 + (q4 & 1) * 4;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int idx = base + i;
      part[warp * 128 + 16 * (idx >> 1) + g8 + 8 * (idx & 1)] = dbp[i];
    }
  }
  __syncthreads();
  for (int o = tid; o < 128; o += C::NTHR) {
    float s = 0.f;
#pragma unroll
    for (int w = 0; w < C::NW; ++w) s += part[w * 128 + o];
    p.db_part[(size_t)blockIdx.x * 128 + o] = s;
  }
}

}  // namespace mp80
