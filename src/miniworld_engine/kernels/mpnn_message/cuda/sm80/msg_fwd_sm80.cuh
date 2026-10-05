// msg_fwd_sm80.cuh -- the fused hidden-message reduction of ProteinMPNN on A100 (sm_80), forward (and the no-grad inference):
//
//   a = bf16(gelu(P))   projected = bf16(a W^T + bias)   hidden = bf16(gelu(projected))   reduced[g, :] = sum_k mask[g, k] hidden[g, k, :] / scale
//
// P [G * 48, 128] bf16 (the rounding points are those of the Triton kernel and of the bf16 PyTorch module), W [128, 128] bf16 [out, in], mask fp32, reduced [G, 128] fp32.
// The GEMM is run TRANSPOSED: C^T[o][n] = sum_i W[o][i] a[n][i], so that W is the A operand (its row-major [out, in] layout is the fragment layout: ldmatrix from shared memory) and the activation tile is
// the B operand straight from the tile in registers, and the sum over the 48 neighbours n is a sum over the columns of C^T: thread-local over a lane's columns and ONE quad shuffle (2 xor steps) per group,
// not a 3-level cross-lane reduction.  The kernel is bound by the two GELUs (2 x 128 per row; 9 FMA-pipe + 2 MUFU instructions each, see mpnn_common.cuh) -- not by the tensor cores or HBM.
//
// One warp = one group at a time (three 16-row tiles), no inter-warp traffic: a CTA of 16 warps keeps the 32 KB weight resident in shared memory (swizzled, one copy for the 16 warps); every warp owns a 4 KB
// tile buffer (cp.async, single buffered: the tile is consumed into the B-fragment registers by 8 ldmatrix before the next tile's copy is issued, which then overlaps this tile's whole compute).
#pragma once
#include "mpnn_common.cuh"

namespace mp80 {

struct MsgFwdParams {
  const bf* p;            // [groups * 48][128]
  const bf* w;            // [128][128]
  const bf* bias;         // [128]
  const float* mask;      // [groups * 48]
  float* out;             // [groups][128]
  int64_t groups;
  float inv_scale;
};

struct MsgFwdCfg {
  static constexpr int NW = 16, NTHR = NW * 32, MINB = 1;
  static constexpr int W_BYTES = 128 * 256, TILE = 16 * 256, WSTRIDE = TILE + 128;       // + 16 mask floats (64 B), padded
  static constexpr int SMEM = W_BYTES + NW * WSTRIDE;
};

__global__ void __launch_bounds__(MsgFwdCfg::NTHR, MsgFwdCfg::MINB) msg_fwd_kernel(const MsgFwdParams p) {
  using C = MsgFwdCfg;
  extern __shared__ __align__(128) unsigned char smem_raw[];
  const uint32_t sb = smem_u32(smem_raw);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g8 = lane >> 2, q4 = lane & 3;

  for (int i = tid; i < 2048; i += C::NTHR) {                      // W -> shared memory, swizzled rows of 256 B
    const int r = i >> 4, g = i & 15;
    cp_async16(sb + sw256(r, g), p.w + r * 128 + g * 8);
  }
  cp_async_commit();

  const int64_t total_warps = (int64_t)gridDim.x * C::NW;
  const int64_t gw = (int64_t)blockIdx.x * C::NW + warp;
  const int64_t ng = gw < p.groups ? (p.groups - gw + total_warps - 1) / total_warps : 0;
  const int64_t ntiles = ng * 3;
  const uint32_t tb = sb + C::W_BYTES + warp * C::WSTRIDE;

  // per-lane ldmatrix addressing.  A fragments (W, rows of 16 outputs, k step s): matrix mi = lane >> 3 -> row (mi & 1) * 8 + (lane & 7), granule 2 s + (mi >> 1);
  // B fragments (tile rows n, k step s): matrix mi -> tile row (mi >> 1) * 8 + (lane & 7), granule 2 s + (mi & 1).  Both rows are 256 B swizzled by (row & 7) = lane & 7.
  const int mi = lane >> 3, x7 = lane & 7;
  const uint32_t abase = sb + ((mi & 1) * 8 + x7) * 256;
  const uint32_t bbase = tb + ((mi >> 1) * 8 + x7) * 256;
  const int ga = mi >> 1, gb = mi & 1;

  float bz[8][2];                                                  // bias[16 m + g8 + 8 h]
#pragma unroll
  for (int m = 0; m < 8; ++m)
#pragma unroll
    for (int h = 0; h < 2; ++h) bz[m][h] = __bfloat162float(p.bias[16 * m + g8 + 8 * h]);

  auto tile_row0 = [&](int64_t k) { return ((gw + (k / 3) * total_warps) * 3 + k % 3) * 16; };
  auto issue = [&](int64_t k) {
    const int64_t r0 = tile_row0(k);
    const bf* src = p.p + r0 * 128;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int c = lane + 32 * i;
      cp_async16(tb + sw256(c >> 4, c & 15), src + c * 8);
    }
    if (lane < 4) cp_async16(tb + C::TILE + lane * 16, p.mask + r0 + lane * 4);
  };

  cp_async_wait<0>();                                              // W
  __syncthreads();
  if (ntiles > 0) issue(0);
  cp_async_commit();

  float sums[8][2];
#pragma unroll
  for (int m = 0; m < 8; ++m) { sums[m][0] = 0.f; sums[m][1] = 0.f; }
  int t3 = 0;
  for (int64_t k = 0; k < ntiles; ++k) {
    cp_async_wait<0>();
    __syncwarp();
    uint32_t bfr[8][4];                                            // GELU'd activation, B fragments: [k step][n tile 0: b0, b1; n tile 1: b0, b1]
#pragma unroll
    for (int s = 0; s < 8; ++s) ldsm_x4(bfr[s], bbase + ((((2 * s + gb) ^ x7)) << 4));
    const uint2 m0 = lds64(tb + C::TILE + (2 * q4) * 4), m1 = lds64(tb + C::TILE + (8 + 2 * q4) * 4);
    const float mk[2][2] = {{__uint_as_float(m0.x), __uint_as_float(m0.y)}, {__uint_as_float(m1.x), __uint_as_float(m1.y)}};
    __syncwarp();                                                  // every lane has read the tile: the next tile's copy may overwrite it
    if (k + 1 < ntiles) issue(k + 1);
    cp_async_commit();

#pragma unroll
    for (int s = 0; s < 8; ++s)
#pragma unroll
      for (int i = 0; i < 4; ++i) bfr[s][i] = pack_bf16(gelu_f(bf16lo(bfr[s][i])), gelu_f(bf16hi(bfr[s][i])));

    // two output m-tiles at a time: four independent accumulator chains per k step (the chains of one tile alone are 8 dependent mma each)
#pragma unroll
    for (int mp = 0; mp < 4; ++mp) {
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
        for (int j = 0; j < 2; ++j)
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            const uint32_t pr = pack_bf16(acc[u][j][2 * h] + bz[m][h], acc[u][j][2 * h + 1] + bz[m][h]);   // projected = bf16(acc + bias)
            const uint32_t hr = pack_bf16(gelu_f(bf16lo(pr)), gelu_f(bf16hi(pr)));                          // hidden = bf16(gelu(projected))
            sums[m][h] = fmaf(bf16lo(hr), mk[j][0], sums[m][h]);
            sums[m][h] = fmaf(bf16hi(hr), mk[j][1], sums[m][h]);
          }
      }
    }

    if (++t3 == 3) {                                               // the group is complete: sum over the quad, scale, store
      t3 = 0;
      const int64_t g = gw + (k / 3) * total_warps;
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
