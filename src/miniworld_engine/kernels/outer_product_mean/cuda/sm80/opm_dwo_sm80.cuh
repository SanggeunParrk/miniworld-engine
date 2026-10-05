// opm_dwo_sm80.cuh -- OuterProductMean backward: the weight gradient of the output projection straight off the kept grouped outer product.
//
//   dWo[z, (c, e)] = sum_(i, j) dzn[i, j, z] O[(i, c), (j, e)]
//
// a GEMM with M = d_pair (z), N = 1024 (c, e), K = the L^2 pairs, so the pair axis is split over CTAs (rows of i) and the fp32 partials are summed
// afterwards in a fixed order (`reduce_rows`): no atomics, bit-reproducible.  O is read in place (no permuted copy).  Grid = (8 column tiles of 128 =
// four c values) x (d_pair / 128 row tiles of z) x (splits).  A stage is 32 pairs (one i, 32 consecutive j): A = dzn[pairs][128 z] (256-byte
// rows, 8 KiB), B = O[pairs][128 (c, e)] gathered from four O rows (8 KiB); both are K-major in shared memory (rows = pairs), so both fragments come
// through ldmatrix.trans.  Warp tile 64 x 32 (2 x 4 warps), a 4-stage cp.async ring, one barrier per stage.
#pragma once
#include "sm80_common.cuh"

namespace opm80 {

struct DwoParams {
  const __nv_bfloat16* dzn;      // [L][L][CZ]
  const __nv_bfloat16* O;        // [L * 32][ldo]
  float* part;                   // [splits][CZ][1024]
  long ldo;
  int L, CZ, i_per;
};

struct DwoCfg {
  static constexpr int NTHR = 256, STAGES = 4, STAGE = 2 * 32 * 256, SMEM = STAGES * STAGE;   // 64 KiB
};

DEVI void dwo_load(const DwoParams& p, uint32_t sA, uint32_t sB, int i, int j0, int n0, int c0, int tid) {
  // A: dzn[(i, j0 + r)][n0 .. n0 + 127]: 16 pieces of 16 B per pair row
#pragma unroll
  for (int it = 0; it < 2; ++it) {
    const int u = it * 256 + tid, r = u >> 4, pc = u & 15;
    const bool ok = j0 + r < p.L;
    const __nv_bfloat16* src = p.dzn + ((long)i * p.L + (ok ? j0 + r : 0)) * p.CZ + n0 + pc * 8;
    cp_async16(sA + swzn<16>(r, pc), src, ok ? 16u : 0u);
  }
  // B: O[(i, c0 + cc)][(j0 + r) * 32 + e8 * 8 ..]: a warp = 8 pairs x 4 chunks of one c row (512 contiguous bytes)
  {
    const int e8 = (tid >> 3) & 3, r = (tid & 7) + 8 * ((tid >> 5) & 3);
#pragma unroll
    for (int it = 0; it < 2; ++it) {
      const int cc = (tid >> 7) | (it << 1);
      const bool ok = j0 + r < p.L;
      const __nv_bfloat16* src = p.O + (long)(i * 32 + c0 + cc) * p.ldo + (long)(ok ? j0 + r : 0) * 32 + e8 * 8;
      cp_async16(sB + swzn<16>(r, cc * 4 + e8), src, ok ? 16u : 0u);
    }
  }
}

__global__ void __launch_bounds__(256, 2) opm_dwo_kernel(const DwoParams p) {
  extern __shared__ __align__(128) unsigned char smem[];
  const uint32_t sbase = smem_u32(smem);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int wm = warp & 1, wn = warp >> 1;
  const int c0 = blockIdx.x * 4, n0 = blockIdx.y * 128, split = blockIdx.z;
  const int ia = split * p.i_per, ib = min(p.L, ia + p.i_per);
  const int jblocks = (p.L + 31) / 32;
  const int nst = (ib - ia) * jblocks;                      // stages of this CTA (may be <= 0 for the last split)

  float acc[4][4][4];
#pragma unroll
  for (int mt = 0; mt < 4; ++mt)
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) acc[mt][nt][0] = acc[mt][nt][1] = acc[mt][nt][2] = acc[mt][nt][3] = 0.f;

  for (int s = 0; s < DwoCfg::STAGES - 1; ++s) {
    if (s < nst) dwo_load(p, sbase + s * DwoCfg::STAGE, sbase + s * DwoCfg::STAGE + 8192, ia + s / jblocks, (s % jblocks) * 32, n0, c0, tid);
    cp_async_commit();
  }
#pragma unroll 1
  for (int st = 0; st < nst; ++st) {
    cp_async_wait<DwoCfg::STAGES - 2>();
    __syncthreads();
    {
      const int nx = st + DwoCfg::STAGES - 1;
      if (nx < nst) dwo_load(p, sbase + (nx % DwoCfg::STAGES) * DwoCfg::STAGE, sbase + (nx % DwoCfg::STAGES) * DwoCfg::STAGE + 8192, ia + nx / jblocks,
                              (nx % jblocks) * 32, n0, c0, tid);
      cp_async_commit();
    }
    const uint32_t sA = sbase + (st % DwoCfg::STAGES) * DwoCfg::STAGE, sB = sA + 8192;
#pragma unroll
    for (int ks = 0; ks < 2; ++ks) {
      uint32_t af[4][4];
#pragma unroll
      for (int mt = 0; mt < 4; ++mt)
        ldsm_x4_t(af[mt], sA + swzn<16>(16 * ks + (lane & 7) + ((lane >> 4) << 3), 8 * wm + 2 * mt + ((lane >> 3) & 1)));
#pragma unroll
      for (int np = 0; np < 2; ++np) {
        uint32_t bf[4];
        ldsm_x4_t(bf, sB + swzn<16>(16 * ks + (lane & 7) + (((lane >> 3) & 1) << 3), 4 * wn + 2 * np + (lane >> 4)));
#pragma unroll
        for (int mt = 0; mt < 4; ++mt) {
          mma16816(acc[mt][2 * np], af[mt], bf[0], bf[1]);
          mma16816(acc[mt][2 * np + 1], af[mt], bf[2], bf[3]);
        }
      }
    }
  }
  cp_async_wait<0>();

  // partial dWo [128 z][128 columns] of this split: fp32 [split][CZ][1024]
  const int gq = lane >> 2, q = lane & 3;
  float* base = p.part + (long)split * p.CZ * 1024;
#pragma unroll
  for (int mt = 0; mt < 4; ++mt) {
    const int z0 = n0 + wm * 64 + mt * 16 + gq;
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) {
      const int col = c0 * 32 + wn * 32 + nt * 8 + 2 * q;
      stg64(base + (long)z0 * 1024 + col, make_float2(acc[mt][nt][0], acc[mt][nt][1]));
      stg64(base + (long)(z0 + 8) * 1024 + col, make_float2(acc[mt][nt][2], acc[mt][nt][3]));
    }
  }
}

}  // namespace opm80
