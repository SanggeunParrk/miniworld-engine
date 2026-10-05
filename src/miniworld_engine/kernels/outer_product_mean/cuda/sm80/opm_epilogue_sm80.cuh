// opm_epilogue_sm80.cuh -- OuterProductMean forward, back: the grouped outer product O[(i, c), (j, e)] (cuBLAS, bf16) -> the pair update
//
//   out[i, j, :] = bf16( bf16( sum_(c, e) O[(i, c), (j, e)] Wo[:, (c, e)] / n_ij + bias ) + residual[i, j, :] )
//
// i.e. the [i, j, c, e] -> [(i, j), (c, e)] permute, the division by the mask count n_ij (in the fp32 accumulator: the projection is linear, so
// (O / n) Wo == (O Wo) / n, and one bf16 rounding of the module's `out / norm` goes away), the d_hidden^2 -> d_pair projection, the bias and the
// optional pair residual in ONE pass over O (the module spends four).  normalize_before_proj = false (ESMFold2) divides after the bias.
//
// A CTA owns BI i x BJ = 32 j pairs (M = BI * 32 rows) and a column tile of CZ outputs (all of them at d_pair 128); K = 1024 = (c, e) runs in 16 chunks of 64 = two c values.  The A tile
// of a chunk is the pair-major view of O: for one (i, c) the 32 j x 32 e values are 2 KiB contiguous in O (cp.async, 16 B, one warp = 8 pairs =
// 512 contiguous bytes), the B tile is the chunk's 64 columns of Wo (rows of 128 B, L2 resident).  Both are 128-byte rows with the XOR swizzle;
// mma.sync m16n8k16, a multi-stage cp.async ring and one barrier per chunk.  The accumulators are staged as a bf16 [pairs][d_pair] tile in shared
// memory and stored (with the residual) as 16-byte vectors along the pair rows: out[i, j0.., :] is contiguous.
//
// Shapes (EpCfg).  The shipped CTA is 4 i x 32 j = 128 pairs x 128 channels, 8 warps (2 x 4 grid, warp tile 64 x 32), two CTAs per SM (one's store phase hides
// under the other's mma): d_pair 128 is one column tile; at 256 / 384 the column tiles (n0 = 0, 128, ..) are the fastest grid axis, so the 2-3 CTAs that share an A
// tile (the 256 KiB pair-major view of O) run together and read it from L2 after the first.  Kept as variants: WIDE (16 warps in a 4 x 4 grid, warp tile 32 x d_pair / 4,
// all channels, one CTA per SM: the Wo chunk loaded once per 128 pairs; 11-23 % slower than the column tiles) and the first version (8 warps, 64 pairs, all channels).
#pragma once
#include "sm80_common.cuh"

namespace opm80 {

struct EpilogueParams {
  const __nv_bfloat16* O;        // [L * 32][ldo]
  const float* norm;             // [L][L] mask counts (>= 1) or nullptr
  const __nv_bfloat16* wo;       // [CZ][1024]
  const float* bias;             // [CZ]
  const __nv_bfloat16* residual; // [L][L][CZ] or nullptr
  __nv_bfloat16* out;            // [L][L][CZ]
  long ldo;
  int L;
  float norm_const;              // n_ij when `norm` is null
  int norm_first;                // 1: out = acc / n + bias (AF3 order), 0: (acc + bias) / n
  int cz;                        // the full d_pair (a CTA of CZ channels covers the column tile n0 = (blockIdx.x % ntile) CZ of it)
  int ntile;                     // d_pair / CZ column tiles: the fastest grid axis, so the CTAs that share an A tile run together
};

template <int CZ, int NSTAGE = (CZ == 384 ? 2 : 3), int NMINB = 1, bool WIDE = false> struct EpCfg {
  static constexpr int BJ = 32, BI = (CZ <= 128 || WIDE) ? 4 : 2, MT = BI * BJ, NTHR = WIDE ? 512 : 256;     // WIDE: 16 warps, 128 pairs, 4 x 4 warp grid (all channels of d_pair 256 / 384 in one CTA)
  static constexpr int WARPS_M = WIDE ? 4 : 2, WARPS_N = 4;
  static constexpr int WM = MT / WARPS_M, WN = CZ / WARPS_N, RM = WM / 16, RN = WN / 8;
  static constexpr int NCH = CZ / 8;                        // 16-byte chunks per output row
  static constexpr int A_BYTES = MT * 128, B_BYTES = CZ * 128, STAGE = A_BYTES + B_BYTES;
  static constexpr int STAGES = NSTAGE;
  static constexpr int RING = STAGES * STAGE, OUTB = MT * CZ * 2;
  static constexpr int SMEM = (RING > OUTB ? RING : OUTB) + MT * 4 + CZ * 4;
  static constexpr int MINB = NMINB;
  static_assert(RN % 2 == 0, "n8 tiles come in pairs");
  static_assert(SMEM <= 166912, "sm_80 shared memory");
};

template <int CZ, int NSTAGE, int NMINB, bool WIDE>
DEVI void ep_load_stage(const EpilogueParams& p, uint32_t sA, uint32_t sB, int kc, int i0, int j0, int n0, int tid) {
  using C = EpCfg<CZ, NSTAGE, NMINB, WIDE>;
  const int c0 = kc * 2;
  {
    // a warp = 8 consecutive j x 4 chunks of 16 B (512 contiguous bytes of O); consecutive lanes take consecutive rows of the tile, so the eight
    // lanes of a shared-memory write phase fall on eight distinct swizzled bank groups.  16 warps: the c value and the i parity come from the warp index too
    const int e8 = (tid >> 3) & 3, jl = (tid & 7) + 8 * ((tid >> 5) & 3), cc = (tid >> 7) & 1, il0 = WIDE ? (tid >> 8) : 0;
#pragma unroll
    for (int il = il0; il < C::BI; il += (WIDE ? 2 : 1)) {
      const int i = i0 + il, j = j0 + jl;
      const bool ok = (i < p.L) & (j < p.L);
      const __nv_bfloat16* src = p.O + (long)((ok ? i : 0) * 32 + c0 + cc) * p.ldo + (long)(ok ? j : 0) * 32 + e8 * 8;
      cp_async16(sA + swz128(il * 32 + jl, cc * 4 + e8), src, ok ? 16u : 0u);
    }
  }
#pragma unroll
  for (int it = 0; it < CZ * 8 / C::NTHR; ++it) {
    const int u = it * C::NTHR + tid, n = u >> 3, pc = u & 7;
    cp_async16(sB + swz128(n, pc), p.wo + (long)(n0 + n) * 1024 + kc * 64 + pc * 8);
  }
}

template <int CZ, int NSTAGE = (CZ == 384 ? 2 : 3), int NMINB = 1, bool WIDE = false>
__global__ void __launch_bounds__(EpCfg<CZ, NSTAGE, NMINB, WIDE>::NTHR, EpCfg<CZ, NSTAGE, NMINB, WIDE>::MINB) opm_epilogue_kernel(const EpilogueParams p) {
  using C = EpCfg<CZ, NSTAGE, NMINB, WIDE>;
  extern __shared__ __align__(128) unsigned char smem[];
  const uint32_t sbase = smem_u32(smem);
  float* sinv = reinterpret_cast<float*>(smem + (C::RING > C::OUTB ? C::RING : C::OUTB));
  float* sbias = sinv + C::MT;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int wm = warp % C::WARPS_M, wn = warp / C::WARPS_M;
  const int nt = blockIdx.x % p.ntile, n0 = nt * CZ;
  const int i0 = blockIdx.y * C::BI, j0 = (blockIdx.x / p.ntile) * C::BJ;

  for (int s = 0; s < C::STAGES - 1; ++s) {
    ep_load_stage<CZ, NSTAGE, NMINB, WIDE>(p, sbase + s * C::STAGE, sbase + s * C::STAGE + C::A_BYTES, s, i0, j0, n0, tid);
    cp_async_commit();
  }
  if (tid < C::MT) {
    const int il = tid / C::BJ, jl = tid % C::BJ, i = i0 + il, j = j0 + jl;
    float n = p.norm_const;
    if ((i < p.L) & (j < p.L) && p.norm != nullptr) n = p.norm[(long)i * p.L + j];
    sinv[tid] = 1.0f / n;
  }
  for (int v = tid; v < CZ; v += C::NTHR) sbias[v] = p.bias[n0 + v];

  float acc[C::RM][C::RN][4];
#pragma unroll
  for (int mt = 0; mt < C::RM; ++mt)
#pragma unroll
    for (int nt = 0; nt < C::RN; ++nt) acc[mt][nt][0] = acc[mt][nt][1] = acc[mt][nt][2] = acc[mt][nt][3] = 0.f;

#pragma unroll 1
  for (int kc = 0; kc < 16; ++kc) {
    cp_async_wait<C::STAGES - 2>();
    __syncthreads();
    if (kc + C::STAGES - 1 < 16) {
      const int st = (kc + C::STAGES - 1) % C::STAGES;
      ep_load_stage<CZ, NSTAGE, NMINB, WIDE>(p, sbase + st * C::STAGE, sbase + st * C::STAGE + C::A_BYTES, kc + C::STAGES - 1, i0, j0, n0, tid);
    }
    cp_async_commit();
    const uint32_t sA = sbase + (kc % C::STAGES) * C::STAGE, sB = sA + C::A_BYTES;
#pragma unroll
    for (int ks = 0; ks < 4; ++ks) {
      uint32_t af[C::RM][4];
#pragma unroll
      for (int mt = 0; mt < C::RM; ++mt) ldsm_x4(af[mt], sA + swz128(wm * C::WM + mt * 16 + (lane & 15), 2 * ks + (lane >> 4)));
#pragma unroll
      for (int np = 0; np < C::RN / 2; ++np) {
        uint32_t bf[4];
        ldsm_x4(bf, sB + swz128(wn * C::WN + np * 16 + (lane & 7) + ((lane >> 4) << 3), 2 * ks + ((lane >> 3) & 1)));
#pragma unroll
        for (int mt = 0; mt < C::RM; ++mt) {
          mma16816(acc[mt][2 * np], af[mt], bf[0], bf[1]);
          mma16816(acc[mt][2 * np + 1], af[mt], bf[2], bf[3]);
        }
      }
    }
  }
  cp_async_wait<0>();
  __syncthreads();                                            // the ring is dead: it becomes the output tile

  // ---- / n, + bias, bf16, into the [pairs][CZ] tile (swizzled: the eight rows of a warp's store land on eight bank groups)
  {
    const int g = lane >> 2, q = lane & 3;
#pragma unroll
    for (int mt = 0; mt < C::RM; ++mt) {
      const int r0 = wm * C::WM + mt * 16 + g, r1 = r0 + 8;
      const float i0v = sinv[r0], i1v = sinv[r1];
#pragma unroll
      for (int nt = 0; nt < C::RN; ++nt) {
        const int col = wn * C::WN + nt * 8 + 2 * q;
        const float b0 = sbias[col], b1 = sbias[col + 1];
        float v00, v01, v10, v11;
        if (p.norm_first) {
          v00 = fmaf(acc[mt][nt][0], i0v, b0); v01 = fmaf(acc[mt][nt][1], i0v, b1);
          v10 = fmaf(acc[mt][nt][2], i1v, b0); v11 = fmaf(acc[mt][nt][3], i1v, b1);
        } else {
          v00 = (acc[mt][nt][0] + b0) * i0v; v01 = (acc[mt][nt][1] + b1) * i0v;
          v10 = (acc[mt][nt][2] + b0) * i1v; v11 = (acc[mt][nt][3] + b1) * i1v;
        }
        sts32(sbase + swzn<C::NCH>(r0, col >> 3) + (col & 7) * 2, pack_bf16(v00, v01));
        sts32(sbase + swzn<C::NCH>(r1, col >> 3) + (col & 7) * 2, pack_bf16(v10, v11));
      }
    }
  }
  __syncthreads();

  // ---- store: pair rows are contiguous in `out` (j fastest), 16 B per thread; the residual is added after the update's own rounding
#pragma unroll 4
  for (int u = tid; u < C::MT * C::NCH; u += C::NTHR) {
    const int row = u / C::NCH, ch = u % C::NCH;
    const int i = i0 + row / C::BJ, j = j0 + row % C::BJ;
    if ((i < p.L) & (j < p.L)) {
      uint4 v = lds128(sbase + swzn<C::NCH>(row, ch));
      const long off = ((long)i * p.L + j) * p.cz + n0 + ch * 8;
      if (p.residual != nullptr) {
        const uint4 r = ldg128(p.residual + off);
        v.x = add_bf16x2(v.x, r.x); v.y = add_bf16x2(v.y, r.y); v.z = add_bf16x2(v.z, r.z); v.w = add_bf16x2(v.w, r.w);
      }
      stg128(p.out + off, v);
    }
  }
}

}  // namespace opm80
