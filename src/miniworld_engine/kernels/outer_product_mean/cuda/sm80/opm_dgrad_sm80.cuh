// opm_dgrad_sm80.cuh -- OuterProductMean backward, first stage: the gradient of the grouped outer product straight from the pair gradient.
//
//   dzn[i, j, :]      = bf16( dz[i, j, :] / n_ij )                                     (kept for dWo)
//   dO[(i, c), (j, e)] = bf16( sum_z dzn[i, j, z] Wo[z, (c, e)] )                      written in the GROUPED layout of O: the [N, N, 1024] permute
//                                                                                      of the module's chain never exists
//   dbo partials      = sum over the CTA's pairs of dz (AF3 order) or dz / n (ESMFold2 order), one row per CTA (reduced by `reduce_rows`)
//
// A CTA owns 4 i x 32 j pairs (M = 128 rows); the dzn tile ([128][d_pair], converted in flight) stays in shared memory and the CTA sweeps the 1024 (c, e)
// outputs in 8 passes of 128 columns (four c values): per pass K = d_pair in chunks of 32 rows of Wo (cp.async ring), mma.sync with A from the tile
// (ldmatrix) and B = Wo read through ldmatrix.trans (Wo[z][(c, e)] is k-major rows already: no transposed copy of the weight).  A pass ends with
// a bf16 [128 pairs][128 columns] tile in shared memory and 2 KiB contiguous stores per (i, c): dO[(i, c), j0.. * 32 ..] .
#pragma once
#include "sm80_common.cuh"

namespace opm80 {

struct DgradParams {
  const __nv_bfloat16* dz;       // [L][L][CZ]
  const float* norm;             // [L][L] or nullptr
  const __nv_bfloat16* wo;       // [CZ][1024]
  __nv_bfloat16* dO;             // [L * 32][ldo]
  __nv_bfloat16* dzn;            // [L][L][CZ]
  float* dbo_part;               // [gridDim.x * gridDim.y][CZ]
  long ldo;
  int L;
  float norm_const;
  int norm_first;
};

template <int CZ, int NSTAGE = 3, int NMINB = 1> struct DgCfg {
  static constexpr int BJ = 32, BI = 4, MT = BI * BJ, NTHR = 256;   // 128 pairs
  static constexpr int NCH = CZ / 8;                                 // 16-byte chunks per dzn row
  static constexpr int KCH = CZ / 32;                                // 32-row chunks of Wo per pass
  static constexpr int STAGES = NSTAGE, BST = 32 * 256;              // Wo chunk: 32 rows x 128 columns x 2 B
  static constexpr int ATILE = MT * CZ * 2, RING = STAGES * BST, OUTB = MT * 128 * 2;
  static constexpr int PPR = CZ / 8, ROWS_PP = 256 / PPR;            // conversion pass: pieces per row, rows per sweep
  static constexpr int SMEM = ATILE + RING + OUTB;
  static_assert(SMEM <= 166912, "sm_80 shared memory");
  static_assert(ROWS_PP * CZ * 4 <= OUTB, "the dbo reduction tile lives in the output tile");
};

template <int CZ, int NSTAGE, int NMINB>
DEVI void dg_load_b(const DgradParams& p, uint32_t sB, int g, int tid) {   // linear step g = pass * KCH + chunk
  using C = DgCfg<CZ, NSTAGE, NMINB>;
  const int pn = g / C::KCH, kc = g % C::KCH;
#pragma unroll
  for (int it = 0; it < 2; ++it) {
    const int u = it * C::NTHR + tid, kr = u >> 4, pc = u & 15;
    cp_async16(sB + swzn<16>(kr, pc), p.wo + (long)(kc * 32 + kr) * 1024 + pn * 128 + pc * 8);
  }
}

template <int CZ, int NSTAGE = 3, int NMINB = 1>
__global__ void __launch_bounds__(256, NMINB) opm_dgrad_kernel(const DgradParams p) {
  using C = DgCfg<CZ, NSTAGE, NMINB>;
  extern __shared__ __align__(128) unsigned char smem[];
  const uint32_t sA = smem_u32(smem), sR = sA + C::ATILE, sO = sR + C::RING;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int wm = warp & 1, wn = warp >> 1;
  const int i0 = blockIdx.y * C::BI, j0 = blockIdx.x * C::BJ;
  const int blk = blockIdx.y * gridDim.x + blockIdx.x;

  // the first Wo chunks start loading under the conversion pass
  for (int s = 0; s < C::STAGES - 1; ++s) { dg_load_b<CZ, NSTAGE, NMINB>(p, sR + s * C::BST, s, tid); cp_async_commit(); }

  // ---- dzn = bf16(dz / n) into the A tile (and to global), and the dbo partial sums
  {
    float cs[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) cs[k] = 0.f;
    const int piece = tid % C::PPR, r = tid / C::PPR;
    const bool active = tid < C::ROWS_PP * C::PPR;
    if (active) {
      for (int row0 = 0; row0 < C::MT; row0 += C::ROWS_PP) {
        const int row = row0 + r;
        if (row < C::MT) {
          const int i = i0 + row / C::BJ, j = j0 + row % C::BJ;
          const bool valid = (i < p.L) & (j < p.L);
          uint4 raw = make_uint4(0, 0, 0, 0), pk = make_uint4(0, 0, 0, 0);
          if (valid) {
            raw = ldg128(p.dz + ((long)i * p.L + j) * CZ + piece * 8);
            const float n = p.norm != nullptr ? p.norm[(long)i * p.L + j] : p.norm_const;
            const float d[8] = {bf16lo(raw.x), bf16hi(raw.x), bf16lo(raw.y), bf16hi(raw.y), bf16lo(raw.z), bf16hi(raw.z), bf16lo(raw.w), bf16hi(raw.w)};
            float q[8];
#pragma unroll
            for (int k = 0; k < 8; ++k) {
              q[k] = __fdiv_rn(d[k], n);
              cs[k] += p.norm_first ? d[k] : q[k];
            }
            pk.x = pack_bf16(q[0], q[1]); pk.y = pack_bf16(q[2], q[3]); pk.z = pack_bf16(q[4], q[5]); pk.w = pack_bf16(q[6], q[7]);
            stg128(p.dzn + ((long)i * p.L + j) * CZ + piece * 8, pk);
          }
          sts128(sA + swzn<C::NCH>(row, piece), pk);
        }
      }
    }
    // column sums: reduce over the row groups through shared memory (the output tile is free until the first pass ends)
    float* sred = reinterpret_cast<float*>(smem + C::ATILE + C::RING);
    if (active) {
#pragma unroll
      for (int k = 0; k < 8; ++k) sred[r * CZ + piece * 8 + k] = cs[k];
    }
    cp_async_wait<C::STAGES - 2>();
    __syncthreads();
    for (int col = tid; col < CZ; col += C::NTHR) {
      float s = 0.f;
#pragma unroll
      for (int rr = 0; rr < C::ROWS_PP; ++rr) s += sred[rr * CZ + col];
      p.dbo_part[(long)blk * CZ + col] = s;
    }
  }

  float acc[4][4][4];
  constexpr int TOTAL = 8 * C::KCH;
#pragma unroll 1
  for (int g = 0; g < TOTAL; ++g) {
    const int kc = g % C::KCH;
    cp_async_wait<C::STAGES - 2>();
    __syncthreads();
    if (g + C::STAGES - 1 < TOTAL) dg_load_b<CZ, NSTAGE, NMINB>(p, sR + ((g + C::STAGES - 1) % C::STAGES) * C::BST, g + C::STAGES - 1, tid);
    cp_async_commit();
    if (kc == 0) {
#pragma unroll
      for (int mt = 0; mt < 4; ++mt)
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) acc[mt][nt][0] = acc[mt][nt][1] = acc[mt][nt][2] = acc[mt][nt][3] = 0.f;
    }
    const uint32_t sB = sR + (g % C::STAGES) * C::BST;
#pragma unroll
    for (int ks = 0; ks < 2; ++ks) {
      const int kg = 2 * kc + ks;                                        // k16 step along d_pair
      uint32_t af[4][4];
#pragma unroll
      for (int mt = 0; mt < 4; ++mt) ldsm_x4(af[mt], sA + swzn<C::NCH>(wm * 64 + mt * 16 + (lane & 15), 2 * kg + (lane >> 4)));
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
    if (kc == C::KCH - 1) {
      // ---- the pass is done: acc -> bf16 tile [pair][128 columns] -> dO
      const int pn = g / C::KCH;
      {
        const int gq = lane >> 2, q = lane & 3;
#pragma unroll
        for (int mt = 0; mt < 4; ++mt) {
          const int r0 = wm * 64 + mt * 16 + gq, r1 = r0 + 8;
#pragma unroll
          for (int nt = 0; nt < 4; ++nt) {
            const int col = wn * 32 + nt * 8 + 2 * q;
            sts32(sO + swzn<16>(r0, col >> 3) + (col & 7) * 2, pack_bf16(acc[mt][nt][0], acc[mt][nt][1]));
            sts32(sO + swzn<16>(r1, col >> 3) + (col & 7) * 2, pack_bf16(acc[mt][nt][2], acc[mt][nt][3]));
          }
        }
      }
      __syncthreads();
#pragma unroll
      for (int it = 0; it < 8; ++it) {
        const int e8 = (tid >> 3) & 3, jl = (tid & 7) + 8 * ((tid >> 5) & 3), cc = (tid >> 7) | ((it & 1) << 1), il = it >> 1;
        const int i = i0 + il, j = j0 + jl;
        if ((i < p.L) & (j < p.L)) {
          const uint4 v = lds128(sO + swzn<16>(il * 32 + jl, cc * 4 + e8));
          stg128(p.dO + (long)(i * 32 + pn * 4 + cc) * p.ldo + (long)j * 32 + e8 * 8, v);
        }
      }
    }
  }
}

}  // namespace opm80
