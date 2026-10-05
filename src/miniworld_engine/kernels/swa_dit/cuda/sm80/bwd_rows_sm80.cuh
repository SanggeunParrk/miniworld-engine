// bwd_rows_sm80.cuh -- what the row-tiled backward kernels (out-projection, FFN gate / dy, qkvg) share on A100 / sm_80: how a 16-row tile maps to rows of the [N S] sequence and to the rows of the
// modulation, and how the modulation gradient leaves the kernel.
//
//   MODE_SINGLE  every row has its own modulation row (A = N / B = 1, the conditioning per sample): a tile is 16 consecutive atoms of one sample, tiles are numbered with the sample fastest (the warps of
//                a CTA read the same atoms of different samples), and the modulation gradient of a row is stored where it belongs (no accumulation, nothing to zero).
//   MODE_HOIST   the conditioning is shared by the A = N / B samples of a batch element (A a multiple of 16): a tile is ONE atom of 16 samples (the 8 lanes of a quad group g8 and the two row halves
//                hh are 16 samples), so all its rows read the same modulation row and its gradient is the sum over the tile's 16 rows -- a reduction across the lanes of a warp (``reduce_scatter_g8``),
//                no atomics.  Tiles are numbered atom fastest (neighbouring warps read neighbouring rows); the 16-sample blocks write to ``nblk`` separate partial buffers [nblk][B S][768] that
//                the caller adds in a fixed order.
#pragma once
#include "sm80_common.cuh"

namespace sw80 {

constexpr int MODE_SINGLE = 0, MODE_HOIST = 1;

struct RowMap {
  int rr[2];          // row of the [N S] sequence of the tile's rows g8 and g8 + 8
  int nrow[2];        // its sample n (the head-major tensors [N][4][S][32] are indexed by it)
  int sat[2];         // its atom
  int mrow[2];        // its modulation row
  bool rok[2];
  int blk;            // the partial buffer (MODE_HOIST), 0 otherwise
};

// number of tiles of a call
template <int MODE>
DEVI int bwd_ntile(int M, int S, int B) {
  if (MODE == MODE_SINGLE) return ((S + 15) / 16) * (M / S);
  return S * B * (M / S / B / 16);
}

template <int MODE>
DEVI RowMap row_map(int tile, int M, int S, int B, int g8) {
  RowMap m;
  if (MODE == MODE_SINGLE) {
    const int nsamp = M / S, ntile = ((S + 15) / 16) * nsamp, tsi = tile / nsamp, nsm = tile - tsi * nsamp;
    m.blk = 0;
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
      const int s = 16 * tsi + g8 + 8 * hh;
      m.sat[hh] = s;
      m.nrow[hh] = nsm;
      m.rok[hh] = tile < ntile && s < S;
      m.rr[hh] = nsm * S + s;
      m.mrow[hh] = (nsm % B) * S + s;
    }
  } else {
    const int ntile = S * B * (M / S / B / 16), s = tile % S, rest = tile / S, bb = rest % B;
    m.blk = rest / B;
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
      const int n = (16 * m.blk + g8 + 8 * hh) * B + bb;
      m.sat[hh] = s;
      m.nrow[hh] = n;
      m.rok[hh] = tile < ntile;
      m.rr[hh] = n * S + s;
      m.mrow[hh] = bb * S + s;
    }
  }
  return m;
}

// Sums the NV values of every lane over the 8 lanes that share lane & 3 (lane bits 2, 3, 4: the rows g8 of an mma tile) and scatters the sums: after the call v[0 .. NV / 8) of a lane hold the sums of the
// values base .. base + NV / 8 - 1, base = NV / 2 b2 + NV / 4 b3 + NV / 8 b4 (b2, b3, b4 the lane's bits 2, 3, 4).  NV - NV / 8 shuffles per lane.
template <int NV>
DEVI void reduce_scatter_g8(float (&v)[NV], int lane) {
  static_assert(NV % 8 == 0, "NV must be a multiple of 8");
  const bool b2 = (lane & 4) != 0, b3 = (lane & 8) != 0, b4 = (lane & 16) != 0;
#pragma unroll
  for (int i = 0; i < NV / 2; ++i) {
    const float lo = v[i], hi = v[NV / 2 + i];
    v[i] = (b2 ? hi : lo) + __shfl_xor_sync(0xffffffffu, b2 ? lo : hi, 4);
  }
#pragma unroll
  for (int i = 0; i < NV / 4; ++i) {
    const float lo = v[i], hi = v[NV / 4 + i];
    v[i] = (b3 ? hi : lo) + __shfl_xor_sync(0xffffffffu, b3 ? lo : hi, 8);
  }
#pragma unroll
  for (int i = 0; i < NV / 8; ++i) {
    const float lo = v[i], hi = v[NV / 8 + i];
    v[i] = (b4 ? hi : lo) + __shfl_xor_sync(0xffffffffu, b4 ? lo : hi, 16);
  }
}

DEVI int g8_base(int lane, int nv) { return (nv / 2) * ((lane >> 2) & 1) + (nv / 4) * ((lane >> 3) & 1) + (nv / 8) * ((lane >> 4) & 1); }

}  // namespace sw80
