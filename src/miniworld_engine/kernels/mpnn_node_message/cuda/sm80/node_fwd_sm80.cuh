// node_fwd_sm80.cuh -- the fused ProteinMPNN encoder node message on A100 (sm_80), forward and no-grad inference:
//
//   pre = bf16(bf16(q + bf16(E W1e^T)) + nb[idx])    act = bf16(gelu(pre))    hid = bf16(act W2^T + b2)    reduced[g, :] = sum_k mask[g, k] bf16(gelu(hid)) / scale
//
// E [G * K, 128] bf16 edge states, q [G, 128] the query projection (one row per group), nb [NN, 128] the neighbour projection TABLE gathered by idx [G * K] (int64, rows of the table), W1e / W2 [128, 128] bf16
// [out, in], mask fp32 or bf16 [G * K], reduced [G, 128] fp32; K <= 128 (K = 48 is the shipped graph).  The rounding points are the Triton kernel's and the bf16 PyTorch module's.
//
// Per 16-row tile of a group, one warp, no inter-warp traffic:  GEMM 1 (E W1e^T) in the natural orientation, C[n][o] (n = g8 + 8 h, o = 8 j + 2 q4 (+1)): A = the E tile through ldmatrix, B = W1e through
// ldmatrix; the epilogue adds q (a register row), the gathered neighbour row (cp.async-gathered into a shared tile: the table is small and lives in L2) and runs the first GELU, and its packed result IS the
// B-fragment layout of the transposed second GEMM  C^T[o'][n] = sum_o W2[o'][o] act[n][o]  (A = W2 through ldmatrix), whose epilogue (bias, GELU, mask) sums over the columns n -- thread-local plus one quad
// shuffle per group.  Both weights stay in shared memory (64 KB, swizzled, one copy per CTA of 8 warps).  The edge tile of the next tile is requested when GEMM 1 has consumed this one, the gathered neighbour tile
// when the epilogue has; the next tile's indices and mask are read into registers at the start of a tile.
#pragma once
#include "mpnn_common.cuh"

namespace mp80 {

struct NodeFwdParams {
  const bf* edge;         // [G * K][128]
  const bf* query;        // [G][128]
  const bf* nbt;          // [NN][128] neighbour projection table
  const int64_t* idx;     // [G * K]
  const bf* w1;           // [128][128] (edge block of W1)
  const bf* w2;           // [128][128]
  const bf* b2;           // [128]
  const void* mask;       // [G * K] fp32 or bf16
  float* out;             // [G][128]
  int64_t groups;
  int K;
  float inv_scale;
};

#ifndef MP_NODE_FWD_HALVES
#define MP_NODE_FWD_HALVES 1     // 1: GEMM 1 and its epilogue run in two halves of 64 output channels (32 accumulators live instead of 64): the register budget of 12 warps per SM
#endif
#ifndef MP_NODE_FWD_NW
#define MP_NODE_FWD_NW (MP_NODE_FWD_HALVES ? 12 : 8)
#endif

struct NodeFwdCfg {
  static constexpr int NW = MP_NODE_FWD_NW, NTHR = NW * 32, MINB = 1;
  static constexpr int W_BYTES = 2 * 128 * 256, BIAS_BYTES = 512, TILE = 16 * 256, WSTRIDE = 2 * TILE;                 // the two weights | the bias (256 B, padded) | per warp: the edge tile | the gathered neighbour tile
  static constexpr int SMEM = W_BYTES + BIAS_BYTES + NW * WSTRIDE;
};

template <bool MASK_BF16>
__global__ void __launch_bounds__(NodeFwdCfg::NTHR, NodeFwdCfg::MINB) node_fwd_kernel(const NodeFwdParams p) {
  using C = NodeFwdCfg;
  extern __shared__ __align__(128) unsigned char smem_raw[];
  const uint32_t sb = smem_u32(smem_raw);
  const uint32_t sw1 = sb, sw2 = sb + 128 * 256;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g8 = lane >> 2, q4 = lane & 3;
  const int K = p.K, ntl = (K + 15) >> 4;

  for (int i = tid; i < 2 * 2048; i += C::NTHR) {                  // W1e | W2 -> shared memory, swizzled rows of 256 B
    const int w = i >> 11, r = (i >> 4) & 127, g = i & 15;
    cp_async16(sb + w * 32768 + sw256(r, g), (w ? p.w2 : p.w1) + r * 128 + g * 8);
  }
  if (tid < 16) cp_async16(sb + C::W_BYTES + tid * 16, p.b2 + tid * 8);
  cp_async_commit();

  const int64_t total_warps = (int64_t)gridDim.x * C::NW;
  const int64_t gw = (int64_t)blockIdx.x * C::NW + warp;
  const int64_t ng = gw < p.groups ? (p.groups - gw + total_warps - 1) / total_warps : 0;
  const int64_t ntiles = ng * ntl;
  const uint32_t tbe = sb + C::W_BYTES + C::BIAS_BYTES + warp * C::WSTRIDE, tbn = tbe + C::TILE;
  const bf* sb2 = reinterpret_cast<const bf*>(smem_raw + C::W_BYTES);

  const int mi = lane >> 3, x7 = lane & 7;
  const uint32_t ae = tbe + ((mi & 1) * 8 + x7) * 256;               // E tile as A fragments: matrix mi -> rows (mi & 1) * 8 + x7, granule 2 s + (mi >> 1)
  const int gae = mi >> 1;
  const int gb1 = mi & 1, rb1 = mi >> 1;                              // W1e as B fragments: matrix mi -> o rows 8 (jt + (mi >> 1)) + x7, granule 2 s + (mi & 1)
  const uint32_t aw2 = sw2 + ((mi & 1) * 8 + x7) * 256;               // W2 as A fragments: rows 16 m' + (mi & 1) * 8 + x7, granule 2 s' + (mi >> 1)

  // tile k of this warp = tile t of its group number gi: k / ntl by an invariant-divisor multiply (ntl <= 8, k < 2^29: exact; a 64-bit division by a runtime value is a subroutine call)
  const uint32_t ntl_inv = ntl == 1 ? 0u : 0xFFFFFFFFu / (uint32_t)ntl + 1u;
  auto tile_info = [&](int64_t k, int64_t& row0, int& nv, int64_t& g, int& t) {
    const uint32_t ku = (uint32_t)k, giu = ntl == 1 ? ku : __umulhi(ku, ntl_inv);
    const int64_t gi = giu;
    t = (int)(ku - giu * (uint32_t)ntl);
    g = gw + gi * total_warps;
    row0 = g * K + 16 * t;
    nv = min(16, K - 16 * t);
  };
  // the next tile's indices (lane's rows 2 i + (lane >> 4), i < 8) and mask (n = 2 q4 + {0, 1}, 8 + 2 q4 + {0, 1}) into registers
  auto load_idx = [&](int64_t k, int (&ix)[8]) {
    int64_t row0, g; int nv, t;
    tile_info(k, row0, nv, g, t);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int r = 2 * i + (lane >> 4);
      ix[i] = r < nv ? (int)p.idx[row0 + r] : 0;
    }
  };
  auto load_mask = [&](int64_t k, float (&mk)[4]) {
    int64_t row0, g; int nv, t;
    tile_info(k, row0, nv, g, t);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int n = 2 * q4 + (i & 1) + 8 * (i >> 1);
      float v = 0.f;
      if (n < nv) {
        if constexpr (MASK_BF16) v = __bfloat162float(reinterpret_cast<const bf*>(p.mask)[row0 + n]);
        else v = reinterpret_cast<const float*>(p.mask)[row0 + n];
      }
      mk[i] = v;
    }
  };
  auto issue_e = [&](int64_t k) {
    int64_t row0, g; int nv, t;
    tile_info(k, row0, nv, g, t);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int c = lane + 32 * i, r = c >> 4;
      const bool ok = r < nv;
      cp_async16(tbe + sw256(r, c & 15), p.edge + (ok ? (row0 + r) * 128 + (c & 15) * 8 : 0), ok ? 16 : 0);
    }
  };
  auto issue_nb = [&](int64_t k, const int (&ix)[8]) {
    int64_t row0, g; int nv, t;
    tile_info(k, row0, nv, g, t);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int c = lane + 32 * i, r = c >> 4;
      cp_async16(tbn + sw256(r, c & 15), p.nbt + (int64_t)ix[i] * 128 + (c & 15) * 8, r < nv ? 16 : 0);
    }
  };

  cp_async_wait<0>();                                              // the weights
  __syncthreads();
  int ixn[8];
  float mkn[4];
  if (ntiles > 0) {
    load_idx(0, ixn);
    load_mask(0, mkn);
    issue_e(0);
    issue_nb(0, ixn);
  }
  cp_async_commit();

  float sums[8][2];
#pragma unroll
  for (int m = 0; m < 8; ++m) { sums[m][0] = 0.f; sums[m][1] = 0.f; }
  uint32_t qv[16];                                                 // the group's query row: o = 8 j + 2 q4 (+1) as bf16 pairs
  int tg = 0;
  for (int64_t k = 0; k < ntiles; ++k) {
    int64_t row0, g; int nv, t;
    tile_info(k, row0, nv, g, t);
    float mk[4] = {mkn[0], mkn[1], mkn[2], mkn[3]};
    if (tg == 0) {
#pragma unroll
      for (int j = 0; j < 16; ++j) qv[j] = *reinterpret_cast<const uint32_t*>(p.query + g * 128 + 8 * j + 2 * q4);
    }
    if (k + 1 < ntiles) { load_idx(k + 1, ixn); load_mask(k + 1, mkn); }
    cp_async_wait<0>();
    __syncwarp();

    // ---- GEMM 1: C[n][o] = sum_i E[n][i] W1e[o][i]; epilogue 1: pre = bf16(bf16(q + bf16(acc)) + nb), act = bf16(gelu(pre)); act[j][h] = the pair (n = g8 + 8 h, o = 8 j + 2 q4 (+1))
    uint32_t actf[16][2];
#if MP_NODE_FWD_HALVES
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {                               // output channels 64 hh .. 64 hh + 63
      float acc1[8][4];
#pragma unroll
      for (int j = 0; j < 8; ++j) { acc1[j][0] = acc1[j][1] = acc1[j][2] = acc1[j][3] = 0.f; }
#pragma unroll
      for (int s = 0; s < 8; ++s) {
        uint32_t a[4];
        ldsm_x4(a, ae + (((2 * s + gae) ^ x7) << 4));
#pragma unroll
        for (int jp = 0; jp < 4; ++jp) {
          uint32_t b[4];
          ldsm_x4(b, sw1 + (8 * (2 * (4 * hh + jp) + rb1) + x7) * 256 + (((2 * s + gb1) ^ x7) << 4));
          mma16816(acc1[2 * jp], a, b[0], b[1]);
          mma16816(acc1[2 * jp + 1], a, b[2], b[3]);
        }
      }
      if (hh == 1) {
        __syncwarp();                                              // the edge tile has been consumed
        if (k + 1 < ntiles) issue_e(k + 1);
        cp_async_commit();
      }
#pragma unroll
      for (int jj = 0; jj < 8; ++jj) {
        const int j = 8 * hh + jj;
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const uint32_t pr = pack_bf16(acc1[jj][2 * h], acc1[jj][2 * h + 1]);
          const uint32_t nbp = lds32(tbn + sw256(g8 + 8 * h, j) + 4 * q4);
          const uint32_t pre = node_pre(pr, qv[j], nbp);
          actf[j][h] = pack_bf16(gelu_f(bf16lo(pre)), gelu_f(bf16hi(pre)));
        }
      }
    }
#else
    float acc1[16][4];
#pragma unroll
    for (int j = 0; j < 16; ++j) { acc1[j][0] = acc1[j][1] = acc1[j][2] = acc1[j][3] = 0.f; }
#pragma unroll
    for (int s = 0; s < 8; ++s) {
      uint32_t a[4];
      ldsm_x4(a, ae + (((2 * s + gae) ^ x7) << 4));
#pragma unroll
      for (int jp = 0; jp < 8; ++jp) {
        uint32_t b[4];
        ldsm_x4(b, sw1 + (8 * (2 * jp + rb1) + x7) * 256 + (((2 * s + gb1) ^ x7) << 4));
        mma16816(acc1[2 * jp], a, b[0], b[1]);
        mma16816(acc1[2 * jp + 1], a, b[2], b[3]);
      }
    }
    __syncwarp();                                                  // the edge tile has been consumed
    if (k + 1 < ntiles) issue_e(k + 1);
    cp_async_commit();
#pragma unroll
    for (int j = 0; j < 16; ++j)
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const uint32_t pr = pack_bf16(acc1[j][2 * h], acc1[j][2 * h + 1]);
        const uint32_t nbp = lds32(tbn + sw256(g8 + 8 * h, j) + 4 * q4);
        const uint32_t pre = node_pre(pr, qv[j], nbp);
        actf[j][h] = pack_bf16(gelu_f(bf16lo(pre)), gelu_f(bf16hi(pre)));
      }
#endif
    __syncwarp();                                                  // the neighbour tile has been consumed
    if (k + 1 < ntiles) issue_nb(k + 1, ixn);
    cp_async_commit();

    // ---- GEMM 2 (transposed): C^T[o'][n] = sum_o W2[o'][o] act[n][o]; epilogue 2 and the neighbour sums
#pragma unroll
    for (int m = 0; m < 8; ++m) {
      float acc[2][4];
#pragma unroll
      for (int j = 0; j < 2; ++j) { acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f; }
#pragma unroll
      for (int s = 0; s < 8; ++s) {
        uint32_t a[4];
        ldsm_x4(a, aw2 + m * 4096 + (((2 * s + (mi >> 1)) ^ x7) << 4));
        mma16816(acc[0], a, actf[2 * s][0], actf[2 * s + 1][0]);
        mma16816(acc[1], a, actf[2 * s][1], actf[2 * s + 1][1]);
      }
#pragma unroll
      for (int j = 0; j < 2; ++j)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const float bzv = __bfloat162float(sb2[16 * m + g8 + 8 * h]);
          const uint32_t hid = pack_bf16(acc[j][2 * h] + bzv, acc[j][2 * h + 1] + bzv);
          const uint32_t gl = pack_bf16(gelu_f(bf16lo(hid)), gelu_f(bf16hi(hid)));
          sums[m][h] = fmaf(bf16lo(gl), mk[2 * j], sums[m][h]);
          sums[m][h] = fmaf(bf16hi(gl), mk[2 * j + 1], sums[m][h]);
        }
    }

    if (++tg == ntl) {                                             // the group is complete
      tg = 0;
#pragma unroll
      for (int m = 0; m < 8; ++m)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const float v = quad_sum(sums[m][h]);
          if (q4 == 0) p.out[g * 128 + 16 * m + g8 + 8 * h] = v * p.inv_scale;
          sums[m][h] = 0.f;
        }
    }
  }
}

}  // namespace mp80
